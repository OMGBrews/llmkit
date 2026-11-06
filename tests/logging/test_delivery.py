"""Delivery contracts in fresh processes with externally controlled stuck sinks.

Each blocked scenario prints ``ready`` after its sink has entered. The parent
requires a positive ``bounded`` milestone *before* releasing that sink, then
collects the child's assertions and exit status. A missing deadline therefore
fails an identified milestone even when releasing the sink lets the child
recover; the final process watchdog is containment, not the sole oracle.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import TextIO, cast

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_COMMON = r"""
import atexit
import asyncio
import contextvars
import threading
from pathlib import Path
from llmkit import (
    LLMCallRecord, LogDeliveryPolicy, configure_llm_logging,
    get_logging_health, flush_llm_logs, flush_llm_logs_async,
    capture_llm_records, capture_llm_log_paths,
)
from llmkit.capture import record_call_async, submit_record_call
from tests._support import make_record

release = threading.Event()
entered = threading.Event()
threading.Thread(target=lambda: (input(), release.set()), daemon=True).start()
# Let the parent's release reach the stdin reader before interpreter teardown.
atexit.register(release.wait)

async def until(event):
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(.001)

class BlockingSink:
    def __init__(self):
        self.records = []
        self.threads = []
    def write(self, record):
        self.write_returning_path(record)
    def write_returning_path(self, record):
        self.threads.append(threading.get_ident())
        entered.set()
        release.wait()
        self.records.append(record)
        return Path('/delivered') / str(record.label)
"""


def _case(source: str, *, common: bool = True) -> None:
    """Assert progress before release, then require a clean child completion."""
    lines: list[str] = []
    messages: queue.Queue[str | None] = queue.Queue()
    process: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-u", "-c", (_COMMON if common else "") + source],
        cwd=_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    output = cast("TextIO", process.stdout)

    def read() -> None:
        for line in output:
            lines.append(line)
            messages.put(line.rstrip())
        messages.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    def milestone(name: str, timeout: float) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty:
                return False
            if line is None:
                return False
            if line == name:
                return True

    try:
        ready = milestone("ready", 20)
        bounded = ready and milestone("bounded", 1)
        assert process.stdin is not None
        try:
            _ = process.stdin.write("release\n")
            process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            _ = process.wait()
            pytest.fail("child did not finish after sink release:\n" + "".join(lines))
        reader.join(timeout=2)
        assert ready, "sink/setup milestone missing:\n" + "".join(lines)
        assert bounded, "bounded progress did not occur before sink release:\n" + "".join(lines)
        assert code == 0, "child assertions failed:\n" + "".join(lines)
        assert "done\n" in lines, "successful completion marker missing:\n" + "".join(lines)
    finally:
        if process.poll() is None:
            process.kill()
            _ = process.wait()
        output.close()
        if process.stdin is not None:
            process.stdin.close()


def test_capacity_worker_limit_deadline_and_no_late_capture() -> None:
    _case(r"""
sink = BlockingSink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(queue_capacity=2, delivery_wait=.02))
async def main():
    with capture_llm_records() as records, capture_llm_log_paths() as paths:
        first = asyncio.create_task(record_call_async(make_record(label='active')))
        await until(entered)
        print('ready', flush=True)
        rest = [asyncio.create_task(record_call_async(make_record(label=str(i)))) for i in range(12)]
        assert await first is None
        assert await asyncio.gather(*rest) == [None] * 12
        health = get_logging_health()
        assert len(records) == 13
        assert paths == []
        assert (health.queued, health.in_flight, health.dropped) == (2, 1, 10), health
        assert health.wait_timeouts == 3, health
        assert len(sink.threads) == 1, sink.threads
        assert not await flush_llm_logs_async(timeout=0)
        print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    health = get_logging_health()
    assert health.handed_off == 3 and health.failed == 0, health
    assert health.wait_timeouts == 3 and health.dropped == 10, health
    assert health.queued == health.in_flight == 0
    assert len(set(sink.threads)) == 1
    assert paths == [] and len(records) == 13
asyncio.run(main())
print('done', flush=True)
""")


def test_cancellation_ends_wait_without_late_capture_or_executor_work() -> None:
    _case(r"""
from unittest.mock import patch
sink = BlockingSink()
configure_llm_logging(sink)
async def forbidden(*args, **kwargs):
    raise AssertionError('managed logging used the default executor')
async def main():
    with patch('asyncio.to_thread', side_effect=forbidden), capture_llm_records() as records, capture_llm_log_paths() as paths:
        call = asyncio.create_task(record_call_async(make_record()))
        await until(entered)
        print('ready', flush=True)
        call.cancel()
        try:
            await call
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('caller cancellation was swallowed')
        assert len(records) == 1 and paths == []
        print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    assert len(records) == 1 and paths == []
    assert get_logging_health().handed_off == 1
asyncio.run(main())
print('done', flush=True)
""")


def test_snapshot_preserves_nested_containers_cycles_and_context() -> None:
    _case(r"""
from dataclasses import replace
marker = contextvars.ContextVar('marker', default='missing')
class Opaque:
    def __deepcopy__(self, memo):
        raise AssertionError('custom copy hooks must not run')
class CustomList(list):
    pass
class Sink(BlockingSink):
    def __init__(self):
        super().__init__()
        self.contexts = []
    def write_returning_path(self, record):
        from llmkit.capture import _captured_records, _captured_log_paths
        assert _captured_records.get() is None and _captured_log_paths.get() is None
        self.contexts.append(marker.get())
        return super().write_returning_path(record)
sink = Sink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(delivery_wait=0))
async def main():
    submit_record_call(make_record(label='blocker'))
    await until(entered)
    print('ready', flush=True)
    opaque = Opaque()
    custom = CustomList(['shared'])
    nested = {'items': [1], 'tuple': ([2],), 'opaque': opaque, 'custom': custom}
    nested['cycle'] = nested
    nested['shared'] = nested['items']
    cycle_list = []
    cycle_tuple = (cycle_list,)
    cycle_list.append(cycle_tuple)
    nested['tuple_cycle'] = cycle_tuple
    prompt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'before'}]}]
    record = make_record(prompt=prompt, response=nested)
    with capture_llm_records() as records, capture_llm_log_paths() as paths:
        token = marker.set('submission-context')
        assert await record_call_async(record) is None
        marker.reset(token)
        assert records == [record]
        prompt[0]['content'][0]['text'] = 'after'
        nested['items'].append(3)
        nested['tuple'][0].append(4)
        custom.append('changed')
        assert paths == []
        print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    saved = sink.records[1]
    assert saved.prompt[0]['content'][0]['text'] == 'before'
    assert saved.response['items'] == [1]
    assert saved.response['tuple'] == ([2],)
    assert saved.response['cycle'] is saved.response
    assert saved.response['shared'] is saved.response['items']
    assert saved.response['tuple_cycle'][0][0] is saved.response['tuple_cycle']
    assert saved.response['opaque'] is opaque and saved.response['custom'] is custom
    assert saved.response['custom'] == ['shared', 'changed']
    assert sink.contexts == ['missing', 'submission-context']
    assert paths == []
asyncio.run(main())
print('done', flush=True)
""")


def test_queued_records_keep_their_sink_and_health_installation() -> None:
    _case(r"""
import llmkit.logging.registry as registry
first = BlockingSink()
configure_llm_logging(first, delivery_policy=LogDeliveryPolicy(delivery_wait=0))
first_health = registry._installed.health
class NextSink:
    def __init__(self):
        self.records = []
    def write(self, record):
        self.records.append(record)
second = NextSink()
async def main():
    submit_record_call(make_record(label='active'))
    await until(entered)
    print('ready', flush=True)
    submit_record_call(make_record(label='old-queued'))
    configure_llm_logging(second)
    submit_record_call(make_record(label='new-queued'))
    assert get_logging_health().handed_off == 0
    assert get_logging_health().queued == 1
    assert first_health.snapshot().queued == 1
    print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    assert [r.label for r in first.records] == ['active', 'old-queued']
    assert [r.label for r in second.records] == ['new-queued']
    assert first_health.snapshot().handed_off == 2
    assert get_logging_health().handed_off == 1
asyncio.run(main())
print('done', flush=True)
""")


def test_flush_covers_only_accepted_prefix_and_cancellation_only_ends_wait() -> None:
    _case(r"""
second_entered = threading.Event()
second_release = threading.Event()
class Sink(BlockingSink):
    def write_returning_path(self, record):
        if record.label == 'later':
            second_entered.set()
            second_release.wait()
            return Path('/later')
        return super().write_returning_path(record)
sink = Sink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(delivery_wait=0))
async def main():
    submit_record_call(make_record(label='first'))
    await until(entered)
    print('ready', flush=True)
    cancelled = asyncio.create_task(flush_llm_logs_async(timeout=2))
    await asyncio.sleep(0)
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError('flush swallowed cancellation')
    prefix = asyncio.create_task(flush_llm_logs_async(timeout=2))
    await asyncio.sleep(0)
    submit_record_call(make_record(label='later'))
    assert not flush_llm_logs(timeout=0)
    print('bounded', flush=True)
    await until(release)
    await until(second_entered)
    assert await asyncio.wait_for(prefix, .5), 'later admission extended flush prefix'
    assert not await flush_llm_logs_async(timeout=0)
    second_release.set()
    assert await flush_llm_logs_async(timeout=2)
asyncio.run(main())
print('done', flush=True)
""")


@pytest.mark.parametrize("failure", ["ordinary", "fatal"])
def test_raising_sinks_and_worker_termination_settle_health(failure: str) -> None:
    _case(
        "failure = "
        + repr(failure)
        + r"""
class Sink(BlockingSink):
    def write_returning_path(self, record):
        entered.set()
        release.wait()
        if failure == 'fatal':
            raise SystemExit('sink terminated worker')
        raise ValueError('sink rejected record')
sink = Sink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(delivery_wait=0))
async def main():
    submit_record_call(make_record(label='active'))
    await until(entered)
    print('ready', flush=True)
    submit_record_call(make_record(label='queued'))
    print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    health = get_logging_health()
    assert health.queued == health.in_flight == 0, health
    if failure == 'ordinary':
        assert health.failed == 2, health
    else:
        assert health.failed >= 1 and health.dropped >= 1, health
        before = health.unavailable
        submit_record_call(make_record(label='after-death'))
        assert get_logging_health().unavailable == before + 1
        assert flush_llm_logs(timeout=0)
asyncio.run(main())
print('done', flush=True)
"""
    )


def test_policy_validation_disabled_capture_and_fast_path_control() -> None:
    _case(r"""
from dataclasses import FrozenInstanceError
import math
for kwargs in ({'queue_capacity': 0}, {'queue_capacity': True}, {'queue_capacity': 1.5},
               {'delivery_wait': -.1}, {'delivery_wait': .101}, {'delivery_wait': math.inf},
               {'delivery_wait': math.nan}, {'drain_timeout': -1}, {'drain_timeout': math.inf},
               {'drain_timeout': math.nan}):
    try:
        LogDeliveryPolicy(**kwargs)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(('invalid policy accepted', kwargs))
policy = LogDeliveryPolicy()
assert (policy.queue_capacity, policy.delivery_wait, policy.drain_timeout) == (256, .1, 1)
try:
    policy.queue_capacity = 10
except FrozenInstanceError:
    pass
else:
    raise AssertionError('policy was mutable')
configure_llm_logging(None)
class Fast:
    def write(self, record):
        pass
    def write_returning_path(self, record):
        return Path('/fast')
async def main():
    with capture_llm_records() as records, capture_llm_log_paths() as paths:
        assert await record_call_async(make_record()) is None
        assert len(records) == 1 and paths == []
    configure_llm_logging(Fast())
    print('ready', flush=True)
    with capture_llm_records() as records, capture_llm_log_paths() as paths:
        assert await record_call_async(make_record()) == Path('/fast')
        assert paths == [Path('/fast')] and len(records) == 1
    try:
        configure_llm_logging(Fast(), delivery_policy=LogDeliveryPolicy(queue_capacity=1))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError('policy changed after managed admission')
    assert flush_llm_logs(timeout=0)
    for timeout in (-1, float('inf'), float('nan')):
        try:
            flush_llm_logs(timeout=timeout)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError('invalid flush timeout accepted')
    print('bounded', flush=True)
asyncio.run(main())
print('done', flush=True)
""")


def test_reentrant_worker_calls_and_flush_do_not_wait_for_themselves() -> None:
    _case(r"""
import time
from llmkit.sync import run_sync
class Sink:
    def __init__(self):
        self.labels = []
    def write(self, record):
        self.labels.append(record.label)
        if record.label == 'outer':
            start = time.monotonic()
            assert not flush_llm_logs(timeout=1)
            assert run_sync(record_call_async(make_record(label='nested'))) is None
            assert time.monotonic() - start < .08, 'worker waited for its own nested delivery'
            entered.set()
sink = Sink()
configure_llm_logging(sink)
async def main():
    print('ready', flush=True)
    await record_call_async(make_record(label='outer'))
    assert await flush_llm_logs_async(timeout=2)
    assert entered.is_set()
    assert sink.labels == ['outer', 'nested']
    print('bounded', flush=True)
asyncio.run(main())
print('done', flush=True)
""")


@pytest.mark.parametrize("kind", ["live-text", "cached-text", "tools"])
def test_abandoned_streams_submit_without_waiting_or_captured_paths(kind: str) -> None:
    _case(
        "kind = "
        + repr(kind)
        + r"""
from pydantic import BaseModel
from llmkit import InMemoryLLMCache, configure_llm_cache, text_llm_call, text_llm_call_stream, tool_llm_call_stream, ToolDefinition
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.testing import Reply, install
from tests._support import ollama_call_defaults
class Args(BaseModel):
    value: int
async def main():
    with ollama_call_defaults(), install(lambda request: Reply(chunks=['a', 'b'], content='ab')):
        if kind == 'cached-text':
            configure_llm_cache(InMemoryLLMCache())
            assert await text_llm_call('q', feature='test') == 'ab'
        sink = BlockingSink()
        configure_llm_logging(sink)
        submit_record_call(make_record(label='blocker'))
        await until(entered)
        print('ready', flush=True)
        with capture_llm_records() as records, capture_llm_log_paths() as paths:
            if kind == 'tools':
                stream = tool_llm_call_stream('q', [ToolDefinition.from_model('do', Args, 'Do.')], feature='test')
            else:
                stream = text_llm_call_stream('q', feature='test')
            await anext(stream)
            await stream.aclose()
            assert len(records) == 1 and records[0].error == STREAM_ABANDONED_ERROR
            assert records[0].cache_hit == (kind == 'cached-text')
            assert paths == []
            print('bounded', flush=True)
        await until(release)
        assert await flush_llm_logs_async(timeout=2)
        assert paths == [] and len(records) == 1
        assert sink.records[1].error == STREAM_ABANDONED_ERROR
asyncio.run(main())
print('done', flush=True)
"""
    )


def test_async_only_process_exit_spends_one_bounded_drain() -> None:
    # Register before importing llmkit: this runs after its final drain. A
    # blocked atexit hook prevents the milestone until the parent releases it.
    _case(
        r"""
import atexit
import asyncio
import threading
def after_drain():
    print('bounded\ndone', flush=True)
    release.wait()
atexit.register(after_drain)
from llmkit import LogDeliveryPolicy, configure_llm_logging
from llmkit.capture import record_call_async
from tests._support import make_record
release = threading.Event()
entered = threading.Event()
threading.Thread(target=lambda: (input(), release.set()), daemon=True).start()
class Sink:
    def write(self, record):
        entered.set()
        release.wait()
configure_llm_logging(Sink(), delivery_policy=LogDeliveryPolicy(delivery_wait=.01, drain_timeout=.02))
asyncio.run(record_call_async(make_record()))
assert entered.is_set()
print('ready', flush=True)
""",
        common=False,
    )


@pytest.mark.parametrize("kind", ["text", "structured", "tools"])
def test_provider_cancellation_does_not_start_a_fresh_delivery_wait(kind: str) -> None:
    _case(
        "kind = "
        + repr(kind)
        + r"""
import time
from pydantic import BaseModel
from llmkit import text_llm_call, structured_llm_call, tool_llm_call, ToolDefinition
from llmkit.testing import Reply, install
from tests._support import ollama_call_defaults, OkSchema
class Args(BaseModel):
    value: int
async def main():
    with ollama_call_defaults(), install(lambda request: Reply(content='{"ok":true}', delay=60)) as fake:
        sink = BlockingSink()
        configure_llm_logging(sink)
        submit_record_call(make_record(label='blocker'))
        await until(entered)
        print('ready', flush=True)
        with capture_llm_records() as records, capture_llm_log_paths() as paths:
            if kind == 'text':
                coro = text_llm_call('q', feature='test')
            elif kind == 'structured':
                coro = structured_llm_call('q', OkSchema, feature='test')
            else:
                coro = tool_llm_call('q', [ToolDefinition.from_model('do', Args, 'Do.')], feature='test')
            call = asyncio.create_task(coro)
            async with asyncio.timeout(2):
                while not fake.requests:
                    await asyncio.sleep(.001)
            start = time.monotonic()
            call.cancel()
            try:
                await call
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('provider cancellation was swallowed')
            assert time.monotonic() - start < .08, 'cancellation finalizer started a fresh delivery wait'
            assert len(records) == 1 and records[0].error == 'CancelledError'
            assert paths == []
            print('bounded', flush=True)
        await until(release)
        assert await flush_llm_logs_async(timeout=2)
        assert paths == [] and len(records) == 1
asyncio.run(main())
print('done', flush=True)
"""
    )


def test_blocked_logging_preserves_provider_results_and_errors() -> None:
    _case(r"""
from llmkit import text_llm_call, NO_RETRY
from llmkit.testing import Reply, install
from tests._support import ollama_call_defaults
failure = ValueError('original provider error')
def fail(request):
    raise failure
async def main():
    with ollama_call_defaults():
        sink = BlockingSink()
        configure_llm_logging(sink)
        submit_record_call(make_record(label='blocker'))
        await until(entered)
        print('ready', flush=True)
        with capture_llm_records() as records:
            with install(lambda request: Reply.text('answer')):
                assert await text_llm_call('q', feature='test') == 'answer'
            with install(fail):
                try:
                    await text_llm_call('q', feature='test', retry=NO_RETRY)
                except ValueError as exc:
                    assert exc is failure
                else:
                    raise AssertionError('provider error was swallowed')
            assert len(records) == 2
            assert records[0].response == 'answer' and records[0].error is None
            assert records[1].error == 'ValueError: original provider error'
            print('bounded', flush=True)
        await until(release)
        assert await flush_llm_logs_async(timeout=2)
asyncio.run(main())
print('done', flush=True)
""")


def test_concurrent_snapshot_reservations_bound_admission_and_flush_scope() -> None:
    _case(r"""
from unittest.mock import patch
import llmkit.logging._delivery as delivery
snapshot_entered = threading.Event()
snapshot_release = threading.Event()
original = delivery._snapshot_record
def prepare(record):
    if record.label == 'preparing':
        snapshot_entered.set()
        snapshot_release.wait()
    return original(record)
sink = BlockingSink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(queue_capacity=1, delivery_wait=0))
async def main():
    with patch.object(delivery, '_snapshot_record', side_effect=prepare):
        preparing = threading.Thread(target=lambda: submit_record_call(make_record(label='preparing')))
        preparing.start()
        await until(snapshot_entered)
        print('ready', flush=True)
        assert get_logging_health().queued == 1
        submit_record_call(make_record(label='rejected'))
        assert get_logging_health().dropped == 1
        assert await flush_llm_logs_async(timeout=0), 'unpublished reservation entered flush prefix'
        assert not entered.is_set()
        print('bounded', flush=True)
        await until(release)
        snapshot_release.set()
        preparing.join(timeout=2)
        assert not preparing.is_alive()
    assert await flush_llm_logs_async(timeout=2)
    assert [record.label for record in sink.records] == ['preparing']
    assert get_logging_health().queued == get_logging_health().in_flight == 0
asyncio.run(main())
print('done', flush=True)
""")


@pytest.mark.parametrize("failure", ["ordinary", "fatal"])
def test_snapshot_failure_preserves_capture_and_releases_reservation(failure: str) -> None:
    _case(
        "failure = "
        + repr(failure)
        + r"""
from unittest.mock import patch
import llmkit.logging._delivery as delivery
class Sink:
    def __init__(self):
        self.labels = []
    def write(self, record):
        self.labels.append(record.label)
sink = Sink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(queue_capacity=1))
async def main():
    print('ready', flush=True)
    with capture_llm_records() as records:
        with patch.object(delivery, '_snapshot_record', side_effect=(RuntimeError('copy failed') if failure == 'ordinary' else SystemExit('copy interrupted'))):
            assert await record_call_async(make_record(label='failed')) is None
        assert len(records) == 1
        health = get_logging_health()
        assert health.preparation_failed == health.failed == 1
        assert health.queued == health.in_flight == 0
        await record_call_async(make_record(label='success'))
    assert sink.labels == ['success']
    assert get_logging_health().handed_off == 1
    print('bounded', flush=True)
asyncio.run(main())
print('done', flush=True)
"""
    )


def test_blocked_exception_formatting_does_not_lock_caller_health_or_admission() -> None:
    _case(r"""
class SlowDescription(Exception):
    def __str__(self):
        entered.set()
        release.wait()
        return 'sink failed'
class Sink:
    def write(self, record):
        raise SlowDescription()
configure_llm_logging(Sink(), delivery_policy=LogDeliveryPolicy(queue_capacity=1, delivery_wait=.02))
async def main():
    with capture_llm_records() as records:
        first = asyncio.create_task(record_call_async(make_record(label='active')))
        await until(entered)
        print('ready', flush=True)
        assert await first is None
        assert await record_call_async(make_record(label='queued')) is None
        assert await record_call_async(make_record(label='rejected')) is None
        health = get_logging_health()
        assert (health.queued, health.in_flight, health.dropped) == (1, 1, 1), health
        assert len(records) == 3
        print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    assert get_logging_health().failed == 2
asyncio.run(main())
print('done', flush=True)
""")


def test_worker_start_failure_preserves_capture_and_reports_unavailable() -> None:
    _case(r"""
from unittest.mock import patch
class Sink:
    def write(self, record):
        raise AssertionError('unstarted worker invoked sink inline')
configure_llm_logging(Sink())
async def main():
    print('ready', flush=True)
    with capture_llm_records() as records, capture_llm_log_paths() as paths:
        with patch('threading.Thread.start', side_effect=RuntimeError('cannot start thread')):
            assert await record_call_async(make_record(label='startup-failure')) is None
        health = get_logging_health()
        assert health.preparation_failed == health.failed == 1, health
        assert health.queued == health.in_flight == 0
        assert await record_call_async(make_record(label='unavailable')) is None
        health = get_logging_health()
        assert health.unavailable >= 1 and health.dropped >= 1
        assert len(records) == 2 and paths == []
        assert await flush_llm_logs_async(timeout=0)
        print('bounded', flush=True)
asyncio.run(main())
print('done', flush=True)
""")


def test_terminal_drain_drops_queued_work_but_does_not_claim_active_loss() -> None:
    _case(r"""
from llmkit.logging._delivery import shutdown_delivery
sink = BlockingSink()
configure_llm_logging(sink, delivery_policy=LogDeliveryPolicy(delivery_wait=0, drain_timeout=.02))
async def main():
    submit_record_call(make_record(label='active'))
    await until(entered)
    print('ready', flush=True)
    submit_record_call(make_record(label='queued'))
    assert not shutdown_delivery()
    health = get_logging_health()
    assert (health.queued, health.in_flight, health.dropped, health.failed) == (0, 1, 1, 0), health
    submit_record_call(make_record(label='after-terminal'))
    assert get_logging_health().dropped == 2
    assert get_logging_health().unavailable == 2
    print('bounded', flush=True)
    await until(release)
    assert await flush_llm_logs_async(timeout=2)
    assert [record.label for record in sink.records] == ['active']
    health = get_logging_health()
    assert health.handed_off == 1 and health.failed == 0 and health.dropped == 2
asyncio.run(main())
print('done', flush=True)
""")


def test_preparation_rechecks_terminal_state_before_publication() -> None:
    _case(r"""
from unittest.mock import patch
import llmkit.logging._delivery as delivery
original = delivery._snapshot_record
def prepare(record):
    entered.set()
    release.wait()
    return original(record)
class Sink:
    def write(self, record):
        raise AssertionError('post-shutdown preparation was published')
configure_llm_logging(Sink())
async def main():
    with patch.object(delivery, '_snapshot_record', side_effect=prepare):
        preparing = threading.Thread(target=lambda: submit_record_call(make_record()))
        preparing.start()
        await until(entered)
        print('ready', flush=True)
        assert get_logging_health().queued == 1
        assert delivery.shutdown_delivery(), 'unpublished work extended terminal drain'
        print('bounded', flush=True)
        await until(release)
        preparing.join(timeout=2)
        assert not preparing.is_alive()
    health = get_logging_health()
    assert health.queued == health.in_flight == 0
    assert health.dropped == health.unavailable == 1
asyncio.run(main())
print('done', flush=True)
""")


def test_raising_diagnostic_handler_does_not_count_one_write_failure_twice() -> None:
    _case(r"""
import logging
class Handler(logging.Handler):
    def emit(self, record):
        raise RuntimeError('diagnostic handler failed')
class Sink:
    def write(self, record):
        raise ValueError('sink failed')
logger = logging.getLogger('llmkit.logging')
logger.handlers[:] = [Handler()]
logger.propagate = False
logger.setLevel(logging.WARNING)
configure_llm_logging(Sink())
async def main():
    print('ready', flush=True)
    with capture_llm_records() as records:
        assert await record_call_async(make_record()) is None
        assert await flush_llm_logs_async(timeout=1)
        assert len(records) == 1
    health = get_logging_health()
    assert health.failed == 1, 'one failed delivery was counted twice: ' + repr(health)
    submit_record_call(make_record(label='after-death'))
    assert get_logging_health().unavailable == 1
    print('bounded', flush=True)
asyncio.run(main())
print('done', flush=True)
""")
