"""Post-loop turn finalization for ``run_conversation``.

Extracted from ``agent/conversation_loop.py`` as part of the god-file
decomposition campaign (``~/.hermes/plans/god-file-decomposition.md``, Phase 1
step 4 — the post-loop ``TurnFinalizer`` seam). ``run_conversation``'s tail
(everything after the main tool-calling ``while`` loop) is lifted here verbatim:
budget-exhaustion summary, trajectory save, session persist, turn diagnostics,
response transforms, result-dict assembly, steer drain, and the memory/skill
review trigger.

Behavior-neutral: the body is moved unchanged. All ``agent.*`` side effects fire
exactly as before; only the post-loop *locals* are passed in as keyword args, and
the assembled ``result`` dict is returned to ``run_conversation`` which returns it
to the caller. The function is synchronous with a single return — mirroring the
region it replaces (no awaits, no early returns).

Module ``logger`` is imported lazily inside the body (``from
agent.conversation_loop import logger``) so this module never imports
``agent.conversation_loop`` at import time -> no import cycle, and the log records
keep the exact logger name (``"agent.conversation_loop"``).
"""

from __future__ import annotations

import hashlib
import logging
import os

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import _sanitize_surrogates
from agent.partial_result_delivery import (
    format_internal_progress_footer,
    is_internal_progress_only,
)


def _assistant_row_missing_visible_text(msg: dict) -> bool:
    """True when an assistant row has no visible text (blank final or tool-only)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    return not flatten_message_text(msg.get("content")).strip()


def _is_pure_tool_call_tail(msg: dict) -> bool:
    """Assistant row with ``tool_calls`` but no visible text of its own."""
    if not isinstance(msg, dict) or not msg.get("tool_calls"):
        return False
    return _assistant_row_missing_visible_text(msg)


# Verification continuation scaffolding flags: verify-on-stop / pre_verify
# inject a synthetic user nudge to keep the agent going one more turn.
# These nudges must be stripped from returned/live history to avoid
# role-alternation breaks and poisoning the resumed transcript. The
# assistant response is real content and is not flagged. (#65919 §7)
_VERIFICATION_CONTINUATION_FLAGS = (
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_final_candidate_synthetic",
)

_LENGTH_CONTINUATION_FLAGS = (
    "_length_continuation_fragment",
    "_length_continuation_nudge",
)

_DELIVERY_FAILURE_RESPONSE = (
    "The response was not released because its final persistence receipt "
    "could not be verified. Please retry the request."
)


def _record_kanban_budget_exhausted(
    kanban_task: str,
    api_call_count: int,
    max_iterations: int,
    logger: logging.Logger,
) -> None:
    """Record a terminal ``timed_out`` outcome for a kanban worker that
    exhausted its iteration budget.

    This is a bounded fallback (#87096): the CAS invariant in ``_end_run``
    (``WHERE ended_at IS NULL``) guarantees idempotence — if another path
    already closed the run this is a no-op — so it is safe to call from
    multiple exit paths.
    """
    try:
        from hermes_cli import kanban_db as _kb
        _conn = _kb.connect()
        try:
            _kb._record_task_failure(
                _conn,
                kanban_task,
                error=(
                    f"Iteration budget exhausted "
                    f"({api_call_count}/{max_iterations}) — "
                    "task could not complete within the allowed "
                    "iterations"
                ),
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                event_payload_extra={
                    "budget_used": api_call_count,
                    "budget_max": max_iterations,
                },
            )
        finally:
            try:
                _conn.close()
            except Exception:
                pass
    except Exception:
        logger.warning(
            "Failed to record budget-exhausted failure for task %s",
            kanban_task,
            exc_info=True,
        )


def _drop_verification_continuation_scaffolding(messages) -> None:
    """Remove verification-continuation nudge messages from *messages* in place.

    Only the synthetic nudges carry these flags, so this strips just the
    nudges while preserving the real attempted-final-answer that was
    persisted to state.db.
    """
    messages[:] = [
        m for m in messages
        if not (isinstance(m, dict) and any(m.get(f) for f in _VERIFICATION_CONTINUATION_FLAGS))
    ]


def _drop_length_continuation_scaffolding(messages) -> None:
    """Remove provider-length retry fragments before a gated terminal flush."""
    messages[:] = [
        m for m in messages
        if not (
            isinstance(m, dict)
            and any(m.get(flag) for flag in _LENGTH_CONTINUATION_FLAGS)
        )
    ]


def _ensure_authorized_assistant_row(
    agent,
    messages,
    final_response: str,
    *,
    hard_gate: bool,
    replace_blank_tail: bool = False,
) -> None:
    """Make the live and committed assistant tail equal the authorized body."""
    if not isinstance(final_response, str) or not final_response:
        return
    tail = messages[-1] if messages and isinstance(messages[-1], dict) else None
    if tail is None or tail.get("role") != "assistant":
        append_message(messages, {"role": "assistant", "content": final_response})
        return
    if tail.get("content") == final_response:
        return
    if not (
        hard_gate
        or _is_pure_tool_call_tail(tail)
        or (replace_blank_tail and _assistant_row_missing_visible_text(tail))
    ):
        return

    tail["content"] = final_response
    stamp_message_timestamp(tail)
    row_id = tail.get("_row_id")
    session_db = getattr(agent, "_session_db", None)
    updater = getattr(session_db, "update_assistant_message_content", None)
    if (
        hard_gate
        and isinstance(row_id, int)
        and not isinstance(row_id, bool)
        and callable(updater)
    ):
        try:
            if updater(agent.session_id or "", row_id, final_response):
                tail[_DB_PERSISTED_MARKER] = True
                return
        except Exception:
            # The terminal flush is the recoverable fallback. The required
            # receipt still fails closed if it cannot observe this exact row.
            pass
    tail.pop(_DB_PERSISTED_MARKER, None)
    agent._db_flush_scan_prefix = None


def _apply_pre_delivery_transforms(
    agent,
    final_response,
    *,
    interrupted,
    preserved_verification_fallback,
    turn_exit_reason,
    effective_task_id,
    turn_id,
    logger,
):
    """Build the exact body that later gates, persistence, and delivery see."""
    response_transformed = False
    pre_transform_response = None
    if final_response and not interrupted:
        try:
            failed_mutations = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if failed_mutations and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(failed_mutations)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as exc:
            logger.debug("file-mutation verifier footer failed: %s", exc)

    if not interrupted:
        try:
            if agent._turn_completion_explainer_enabled():
                stripped = (final_response or "").strip()
                empty_terminal = stripped == "" or stripped == "(empty)"
                partial_fragment = (
                    not empty_terminal
                    and not preserved_verification_fallback
                    and not str(turn_exit_reason).startswith("text_response")
                    and len(stripped) <= 24
                    and stripped[-1:] not in {
                        ".", "!", "?", "。", "！", "？", "`", ")",
                    }
                )
                if (
                    empty_terminal
                    or partial_fragment
                    or str(turn_exit_reason) == "partial_stream_recovery"
                ):
                    explanation = agent._format_turn_completion_explanation(
                        turn_exit_reason,
                        getattr(agent, "_last_persistence_error_cause", None),
                    )
                    if explanation:
                        final_response = (
                            explanation
                            if empty_terminal
                            else stripped + "\n\n" + explanation
                        )
        except Exception as exc:
            logger.debug("turn-completion explainer failed: %s", exc)

    # A model can terminate normally while returning only its private progress
    # status (for example, "research not complete; 9 steps remain").  That is
    # not a delivery-worthy answer, but it is also not an excuse to run another
    # tool loop or a second control gate.  Preserve the model text and append a
    # bounded, factual recovery note assembled from the tool transcript.
    if not interrupted and is_internal_progress_only(final_response):
        final_response = (
            final_response.rstrip()
            + "\n\n"
            + format_internal_progress_footer(messages)
        )

    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook

            results = invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                task_id=effective_task_id,
                turn_id=turn_id,
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for result in results:
                if isinstance(result, str) and result:
                    pre_transform_response = final_response
                    final_response = result
                    response_transformed = True
                    break
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    return final_response, response_transformed, pre_transform_response


def _compensate_failed_receipt(
    agent, messages, *, message_id: int, expected_content: str,
) -> str:
    """Replace an unsealed persisted body by exact row/content identity."""
    session_db = getattr(agent, "_session_db", None)
    updater = getattr(session_db, "update_assistant_message_content", None)
    if not callable(updater) or not updater(
        agent.session_id or "",
        message_id,
        _DELIVERY_FAILURE_RESPONSE,
        expected_content=expected_content,
        clear_private=True,
    ):
        raise RuntimeError("failed to compensate an unsealed assistant body")

    tail = messages[-1] if messages and isinstance(messages[-1], dict) else None
    if not isinstance(tail, dict) or tail.get("_row_id") != message_id:
        raise RuntimeError("persisted assistant receipt row is no longer the tail")
    tail["content"] = _DELIVERY_FAILURE_RESPONSE
    for key in (
        "reasoning", "reasoning_content", "reasoning_details",
        "codex_reasoning_items", "codex_message_items", "api_content",
    ):
        tail.pop(key, None)
    saver = getattr(agent, "_save_session_log", None)
    if callable(saver):
        saver(messages)
    return _DELIVERY_FAILURE_RESPONSE


def finalize_turn(
    agent,
    *,
    final_response,
    api_call_count,
    interrupted,
    failed,
    messages,
    conversation_history,
    effective_task_id,
    turn_id,
    user_message,
    original_user_message,
    _should_review_memory,
    _turn_exit_reason,
    _pending_verification_response=None,
    _pending_verification_response_previewed=False,
):
    """Run the post-loop finalization and return the turn ``result`` dict.

    Lifted verbatim from ``run_conversation`` (the region after the main agent
    loop). See module docstring.
    """
    from agent.conversation_loop import logger

    budget_exhausted = (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    continuation_budget_exhausted = (
        final_response is None
        and bool(_pending_verification_response)
        and budget_fallback_eligible
    )

    iteration_limit_fallback = False
    preserved_verification_fallback = False
    if continuation_budget_exhausted:
        # A verification/continuation gate deliberately withheld a composed
        # answer, then consumed the remaining budget before producing a newer
        # one. Preserve that exact answer instead of replacing it with another
        # fallible model call. The explicit pending value is the provenance
        # guard: unrelated error/recovery exits can never enter this branch.
        final_response = _pending_verification_response
        # Mark the turn as previewed only when the reused candidate was
        # actually streamed to the user as interim content. (#65919 review:
        # response-loss blocker)
        if _pending_verification_response_previewed:
            agent._response_was_previewed = True
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        iteration_limit_fallback = True
        preserved_verification_fallback = True
    elif final_response is None and budget_fallback_eligible:
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, api_call_count)
        iteration_limit_fallback = True

    if iteration_limit_fallback:
        # If running as a kanban worker, signal the dispatcher that the
        # worker could not complete (rather than treating it as a
        # protocol violation). This applies whether the user-facing fallback
        # came from the summary call or an explicitly pending continuation;
        # both exhausted the task budget and must advance the failure circuit.
        #
        # We route through ``_record_task_failure(outcome="timed_out")``
        # rather than ``kanban_block`` so this counts toward the dispatcher's
        # consecutive-failure circuit breaker (#29747 gap 2).
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            _record_kanban_budget_exhausted(
                _kanban_task, api_call_count, agent.max_iterations, logger,
            )
    elif budget_exhausted:
        # Bounded fallback (#87096): budget was exhausted but none of the
        # normal fallback paths were eligible (interrupted / failed /
        # anomalous exit_reason). If running as a kanban worker we must
        # still record a terminal outcome so the task does not remain in
        # an ambiguous lifecycle state. The worker's run is closed via
        # ``_record_task_failure`` (compare-and-swap receipt path) which
        # is a no-op if another path closed it — the CAS invariant in
        # ``_end_run`` (``WHERE ended_at IS NULL``) guarantees idempotence.
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            _record_kanban_budget_exhausted(
                _kanban_task, api_call_count, agent.max_iterations, logger,
            )

    # Determine if conversation completed successfully
    normal_text_response = str(_turn_exit_reason).startswith("text_response(")
    completed = (
        final_response is not None
        and not failed
        and (
            api_call_count < agent.max_iterations
            or normal_text_response
        )
    )

    # Preflight can seed the display count before the provider receives the
    # request. Roll that estimate back only when an interrupt wins the race
    # before any successful provider response. Compaction state remains owned
    # by the real-usage/post-compaction path, including its ``-1`` sentinel.
    # Guard rules (test-double density on this path is high):
    #  - snapshot is type-pinned to a real int — MagicMock agents auto-create
    #    truthy Mock attributes that must never arm the rollback;
    #  - the received-response flag is pinned to ``is not True`` — its real
    #    domain is True/False, and only a literal True means a provider
    #    response completed;
    #  - the compressor method gets a getattr+callable guard — SimpleNamespace
    #    compressor doubles and plugin context engines lack it.
    _preflight_snapshot = getattr(
        agent, "_turn_preflight_display_snapshot", None
    )
    if (
        interrupted is True
        and isinstance(_preflight_snapshot, int)
        and not isinstance(_preflight_snapshot, bool)
        and getattr(agent, "_turn_received_provider_response", False) is not True
        and getattr(agent, "context_compressor", None) is not None
    ):
        _rollback_fn = getattr(
            agent.context_compressor,
            "rollback_interrupted_preflight_display_tokens",
            None,
        )
        if callable(_rollback_fn):
            _rollback_fn(_preflight_snapshot)

    # Post-loop cleanup must never lose the response.  Trajectory save,
    # resource teardown, and session persistence all touch fallible
    # surfaces — file I/O / JSON serialization (_save_trajectory), remote
    # VM/browser teardown over the network (_cleanup_task_resources), and
    # SQLite writes (_persist_session).  A raise from any of them used to
    # propagate straight out of run_conversation, discarding the partial
    # final_response the caller is waiting for (subprocess wrappers saw an
    # empty stdout with no traceback — #8049).  Each step is now guarded
    # independently so one failure can't skip the others, and any errors
    # are surfaced on the result dict via ``cleanup_errors`` rather than
    # killing the turn.
    _cleanup_errors = []

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    try:
        agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)
    except Exception as _save_err:
        _cleanup_errors.append(f"save_trajectory: {_save_err}")
        logger.error("finalize_turn: _save_trajectory failed: %s", _save_err, exc_info=True)

    # Clean up VM and browser for this task after conversation completes
    try:
        agent._cleanup_task_resources(effective_task_id)
    except Exception as _cleanup_err:
        _cleanup_errors.append(f"cleanup_task_resources: {_cleanup_err}")
        logger.error("finalize_turn: _cleanup_task_resources failed: %s", _cleanup_err, exc_info=True)

    # #95514 can recover a visible answer from the stream even when the
    # terminal provider object is blank. Recover before the required Host
    # gate so that path cannot bypass output authorization.
    _recovered_from_stream = False
    if not interrupted and not failed:
        _streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
        if isinstance(_streamed, str):
            _streamed = _streamed.strip()
        else:
            _streamed = ""
        _final_visible = (
            flatten_message_text(final_response).strip() if final_response else ""
        )
        if not _final_visible and _streamed:
            final_response = _streamed
            _recovered_from_stream = True

    _pre_delivery_body = final_response
    final_response, _response_transformed, _pre_transform_response = (
        _apply_pre_delivery_transforms(
            agent,
            final_response,
            interrupted=interrupted,
            preserved_verification_fallback=preserved_verification_fallback,
            turn_exit_reason=_turn_exit_reason,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            logger=logger,
        )
    )
    if isinstance(final_response, str):
        final_response = _sanitize_surrogates(final_response)

    # The candidate gate sees the fully transformed body exactly once. It may
    # allow or replace it, but can no longer trigger another provider turn.
    _candidate_gate_applied = False
    if final_response and not interrupted:
        from agent.final_candidate_gate import evaluate_final_candidate

        _remaining_iterations = min(
            max(0, agent.max_iterations - api_call_count),
            max(0, int(agent.iteration_budget.remaining)),
        )
        _candidate = evaluate_final_candidate(
            response_text=final_response,
            session_id=agent.session_id or "",
            task_id=effective_task_id,
            turn_id=turn_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            finish_reason=str(_turn_exit_reason),
            iteration=api_call_count,
            max_iterations=agent.max_iterations,
            remaining_iterations=_remaining_iterations,
        )
        if _candidate is not None:
            _candidate_gate_applied = True
            if "content" in _candidate:
                final_response = _sanitize_surrogates(_candidate["content"])

    # A candidate replacement is part of the persisted assistant row, not a
    # post-persistence decoration. Prepare that exact row before the required
    # persistence gate so the Host always authorizes the body it will receipt.
    if final_response and not interrupted and _candidate_gate_applied:
        _ensure_authorized_assistant_row(
            agent,
            messages,
            final_response,
            hard_gate=True,
            replace_blank_tail=_recovered_from_stream,
        )

    # Optional Host-owned persistence gate. Callback and ownership failures
    # are fatal. The Host may omit content/hash when it accepts the exact
    # candidate supplied here; Hermes computes and verifies the real hash.
    _hard_persist_gate_applied = False
    _persist_gate_content_sha256 = None
    if final_response and not interrupted:
        from hermes_cli.lifecycle import invoke_required_hook as _invoke_required_hook

        _candidate_sha256 = hashlib.sha256(final_response.encode("utf-8")).hexdigest()
        _persist_gate = _invoke_required_hook(
            "assistant_persist_gate",
            response_text=final_response,
            response_sha256=_candidate_sha256,
            session_id=agent.session_id or "",
            task_id=effective_task_id,
            turn_id=turn_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
        if _persist_gate is not None:
            if (
                not isinstance(_persist_gate, dict)
                or _persist_gate.get("action") != "ALLOW"
            ):
                raise RuntimeError("assistant persistence aborted by Host gate")
            _gate_content = _persist_gate.get("content", final_response)
            if not isinstance(_gate_content, str) or not _gate_content:
                raise RuntimeError("Host gate returned invalid content")
            final_response = _sanitize_surrogates(_gate_content)
            _persist_gate_content_sha256 = hashlib.sha256(
                final_response.encode("utf-8")
            ).hexdigest()
            _declared_hash = _persist_gate.get("content_sha256")
            if _declared_hash is not None and _declared_hash != _persist_gate_content_sha256:
                raise RuntimeError("Host gate content hash does not match its body")
            _hard_persist_gate_applied = True

    # Persist session to both JSON log and SQLite only after private retry
    # scaffolding has been removed. Otherwise a later user "continue" turn
    # can replay assistant("(empty)") / recovery nudges and fall into the
    # same empty-response loop again.
    _session_persisted = False
    try:
        agent._drop_trailing_empty_response_scaffolding(messages)

        # Drop verification-continuation nudges (synthetic user messages)
        # from the live history before the tail-assistant check — only the
        # nudges need stripping; the assistant candidate persists in
        # state.db. (#65919 §7)
        _drop_verification_continuation_scaffolding(messages)
        if _hard_persist_gate_applied:
            _drop_length_continuation_scaffolding(messages)

        # When the turn was interrupted and the last message is a tool
        # result, append a synthetic assistant message to close the
        # tool-call sequence. Without this, the session persists a
        # ``tool → user`` alternation that strict providers (Gemini,
        # Claude) reject, causing them to hallucinate a continuation of
        # the user's message on the next turn (#48879).
        #
        # ``_drop_trailing_empty_response_scaffolding`` only rewinds the
        # tool tail when an empty-response scaffolding flag is present; a
        # clean ``/stop`` interrupt after a successful tool sets no such
        # flag, so the tool result survives as the tail and we close it
        # here instead. On an interrupt ``final_response`` is typically
        # empty, so fall back to an explicit placeholder rather than
        # persisting an empty-content assistant turn.
        if interrupted:
            from agent.message_sanitization import close_interrupted_tool_sequence
            close_interrupted_tool_sequence(messages, final_response)

        # Some recovery/fallback paths return a real final_response without
        # adding a closing assistant message to the transcript (e.g. the
        # partial-stream and prior-turn-content recovery ``break`` sites in
        # ``conversation_loop``). If persisted as-is, the durable session can
        # end at a tool/user message even though the caller — and the gateway
        # platform — already saw a completed assistant response. The next turn
        # then replays a user-only backlog and the model re-answers every
        # "unanswered" message. Close the durable turn at the source, at the
        # single chokepoint every recovery ``break`` flows through, so the
        # invariant "delivered final_response ⇒ assistant row in transcript"
        # holds regardless of which path produced it. (#43849 / #44100)
        #
        # Compare content (not just role) so a verification candidate that
        # matches the final response is not duplicated at budget
        # exhaustion. (#65919 §7)
        if final_response and not interrupted:
            _ensure_authorized_assistant_row(
                agent,
                messages,
                final_response,
                hard_gate=(
                    _hard_persist_gate_applied
                    or _candidate_gate_applied
                    or final_response != _pre_delivery_body
                ),
                replace_blank_tail=_recovered_from_stream,
            )

        # The model has completed its request, so replace API-local
        # voice/model/skill guidance with the clean user input before writing the
        # final durable snapshot and returning the continuation history. Earlier
        # turn-start flushes use the DB-only override because their messages are
        # still needed for the API request; this finalizer runs after that request
        # is complete (#48677 / #63766).
        _apply_override = getattr(agent, "_apply_persist_user_message_override", None)
        if callable(_apply_override):
            _apply_override(messages)

        # ── Post-turn micro-compaction ────────────────────────────
        # After the assistant response is finalized but before the session is
        # persisted, run micro-compaction to absorb the oldest uncompacted
        # exchange into the rolling summary.  This amortizes compression
        # across turns rather than batching it into one big pause.
        if not interrupted and not failed:
            try:
                _compressor = getattr(agent, "context_compressor", None)
                # Strict `is True` + isinstance gates: plugin context engines
                # (and MagicMock compressors in tests) satisfy getattr/duck
                # checks with truthy auto-attributes — a bare truthiness check
                # here called _micro_compact on a mock and spliced its (empty-
                # iterating) return value over the transcript, wiping it.
                if (
                    _compressor
                    and getattr(_compressor, '_micro_compact_enabled', False) is True
                    and callable(getattr(_compressor, '_micro_compact', None))
                    and final_response
                    and not _hard_persist_gate_applied
                    and not _candidate_gate_applied
                    # compression.checkpoint_required: agent init already
                    # forces _micro_compact_enabled off, but the compressor
                    # attribute is plain state a future path could flip on a
                    # live agent. Micro-compaction has no checkpoint hook in
                    # its path, so it must never run while the gate is armed.
                    and getattr(
                        agent, "compression_checkpoint_required", False
                    ) is not True
                    # Persistence-isolated agents (background review fork)
                    # must not micro-compact: the pass burns a real aux-LLM
                    # call on a throwaway replay transcript, and if the
                    # compressor ever holds a session_db binding it would
                    # archive_and_compact the CANONICAL session rows — the
                    # exact write class _persist_disabled exists to stop.
                    and not getattr(agent, "_persist_disabled", False)
                ):
                    _before = len(messages)
                    _compacted = _compressor._micro_compact(messages)
                    # Micro-compaction defrag rewrites the newest MICRO
                    # marker's content and pops _db_persisted from the live
                    # dict in place — the sibling of the pop site above. The
                    # compressor has no agent reference, so it raises a flag
                    # for us to invalidate the bounded flush-scan cursor;
                    # otherwise the rewritten marker row is identity-skipped
                    # and the stale summary persists to state.db.
                    if getattr(
                        _compressor, "_flush_scan_cursor_invalidated", False
                    ):
                        _compressor._flush_scan_cursor_invalidated = False
                        agent._db_flush_scan_prefix = None
                    if isinstance(_compacted, list) and _compacted:
                        messages[:] = _compacted
                    _after = len(messages)
                    if _before != _after:
                        logger.info(
                            "Micro-compaction: %d -> %d messages",
                            _before, _after,
                        )
            except Exception as _mc_err:
                logger.info("Micro-compaction failed: %s", _mc_err)

        agent._persist_session(messages, conversation_history)
        _session_persisted = True
    except Exception as _persist_err:
        _cleanup_errors.append(f"persist_session: {_persist_err}")
        logger.error("finalize_turn: _persist_session failed: %s", _persist_err, exc_info=True)

    if _hard_persist_gate_applied:
        if not _session_persisted:
            raise RuntimeError("authorized assistant content was not persisted")
        _persisted_tail = messages[-1] if messages else None
        _message_id = (
            _persisted_tail.get("_row_id")
            if isinstance(_persisted_tail, dict)
            else None
        )
        _persisted_content = (
            _persisted_tail.get("content")
            if isinstance(_persisted_tail, dict)
            else None
        )
        if (
            not isinstance(_message_id, int)
            or isinstance(_message_id, bool)
            or not isinstance(_persisted_content, str)
            or _persisted_content != final_response
        ):
            raise RuntimeError("Session persistence returned no exact assistant receipt")
        _persisted_sha256 = hashlib.sha256(
            _persisted_content.encode("utf-8")
        ).hexdigest()
        if _persisted_sha256 != _persist_gate_content_sha256:
            raise RuntimeError("persisted assistant body differs from authorized hash")
        _message_id_text = str(_message_id)
        _host_receipt = {
            "message_id": _message_id_text,
            "content_sha256": _persisted_sha256,
            "committed": True,
        }
        try:
            _receipt = _invoke_required_hook(
                "assistant_persist_receipt",
                session_id=agent.session_id or "",
                task_id=effective_task_id,
                turn_id=turn_id,
                message_id=_message_id_text,
                response_text=final_response,
                response_sha256=_persisted_sha256,
                receipt=_host_receipt,
                assistant_response=final_response,
                content_sha256=_persisted_sha256,
                persisted_message_id=_message_id_text,
                persisted_content=_persisted_content,
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            if (
                not isinstance(_receipt, dict)
                or _receipt.get("action") != "COMMITTED"
            ):
                raise RuntimeError(
                    "Host did not acknowledge the assistant persistence receipt"
                )
        except Exception as _receipt_err:
            final_response = _compensate_failed_receipt(
                agent,
                messages,
                message_id=_message_id,
                expected_content=_persisted_content,
            )
            failed = True
            completed = False
            _turn_exit_reason = "assistant_persist_receipt_failed"
            _cleanup_errors.append(f"persist_receipt: {_receipt_err}")
            logger.error(
                "finalize_turn: assistant persistence receipt failed: %s",
                _receipt_err,
                exc_info=True,
            )

    # The gateway owns a separate in-memory history snapshot. Keep it current
    # even when finalization reports a cleanup error: a later prompt must not be
    # sent with the pre-turn snapshot while the durable DB already has this turn.
    try:
        agent._session_messages = messages
    except Exception:
        pass

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # Plugin hook: post_llm_call
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can use this to persist conversation data (e.g. sync
    # to an external memory system).
    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                task_id=effective_task_id,
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Context engine observation hook: notify the active engine that this
    # turn has finished, with the finalized transcript. Complements the
    # per-request select_context() hook (selection before the request;
    # observation after the turn). No-op default, fail-open.
    try:
        from agent.conversation_loop import _notify_context_engine_turn_complete
        # Forward the turn's canonical usage when the host has it. The loop
        # stashes the most recent API response's usage dict (the same
        # canonical buckets fed to ``update_from_response``) on the agent as
        # ``_last_turn_usage``. It is ``None`` on turns that never reached a
        # provider response (early failure / interrupt), which is exactly the
        # contract: real usage when available, ``None`` otherwise.
        _turn_usage = getattr(agent, "_last_turn_usage", None)
        _notify_context_engine_turn_complete(
            agent,
            messages,
            usage=_turn_usage,
            logger=logger,
            turn_id=turn_id,
            task_id=effective_task_id,
            api_call_count=api_call_count,
            interrupted=interrupted,
            failed=failed,
            turn_exit_reason=_turn_exit_reason,
        )
    except Exception as exc:
        logger.warning("on_turn_complete notification failed: %s", exc)

    # Extract reasoning from the CURRENT turn only.  Walk backwards
    # but stop at the user message that started this turn — anything
    # earlier is from a prior turn and must not leak into the reasoning
    # box (confusing stale display; #17055).  Within the current turn
    # we still want the *most recent* non-empty reasoning: many
    # providers (Claude thinking, DeepSeek v4, Codex Responses) emit
    # reasoning on the tool-call step and leave the final-answer step
    # with reasoning=None, so picking only the last assistant would
    # silently drop legitimate same-turn reasoning.
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Class-level surrogate chokepoint (#80366, #55143, #55309, #19819):
    # ``final_response`` is often the RAW SDK content
    # (``assistant_message.content``), not the sanitized copy stored in
    # history by ``build_assistant_message``. Any lone UTF-16 surrogate
    # (U+D800–U+DFFF) in it crashes downstream consumers — oneshot stdout
    # writes, Telegram's ``utf16_len`` length check, Signal formatting,
    # JSON envelope encodes — on every provider (Ollama, NVIDIA NIM, …).
    # Scrub once here, where model text leaves the conversation loop, so
    # every delivery surface receives valid Unicode.
    if isinstance(final_response, str):
        final_response = _sanitize_surrogates(final_response)

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "pre_transform_response": _pre_transform_response,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        # Requested service tier (from request_overrides.extra_body), for
        # billing audits by callers like `hermes -z --usage-file`.
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # Persistence failures already set failed=True + an explanation in
    # final_response; also stamp `error` so gateway surfaces status="error"
    # (and desktop can toast the cause) instead of a quiet complete frame.
    if failed and str(_turn_exit_reason) == "session_persistence_failed":
        result["error"] = final_response or (
            "session storage could not be written — check the state database "
            "health (`hermes doctor`), then send your message again"
        )
        # Machine-readable cause for the gateway/desktop: exactly
        # 'session_persistence_failed:<locked|compression|turn_lease|corrupt|disk|unknown>'.
        # Never clobber a failure_reason another path already stamped.
        if "failure_reason" not in result:
            _cause = getattr(agent, "_last_persistence_error_cause", None)
            result["failure_reason"] = (
                "session_persistence_failed:" + (_cause or "unknown")
            )
    # Surface any post-loop cleanup failures so the caller can distinguish a
    # clean turn from one whose trajectory/session/resource teardown raised
    # (the response is still returned either way — #8049).
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False

    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # Background memory/skill review — runs AFTER the response is delivered
    # so it never competes with the user's task for model attention.
    # Suppressed when skip_background_review=True (e.g. cron) — review forks
    # spawn another AIAgent (~30K tokens / event) and cron sessions have no
    # human-in-the-loop benefit from the review.
    if (
        final_response
        and not interrupted
        and not getattr(agent, "skip_background_review", False)
        and (_should_review_memory or _should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).

    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            completed=completed,
            failed=failed,
            interrupted=interrupted,
            turn_exit_reason=_turn_exit_reason,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False

    return result
