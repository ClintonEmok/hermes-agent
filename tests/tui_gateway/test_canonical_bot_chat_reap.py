"""Tests for #92687: the ws-orphan reaper must not end the canonical Bot Chat.

Bot Mode's forever-chat is identified by ``(profile_name,
title='Bot Chat')``, not by stored session ids. When the desktop's WS drops
for the reap grace window, ``_finalize_session(end_reason='ws_orphan_reap')``
used to end that row — archiving it out from under every future open. The
plugin's recreate then collides with the global title UNIQUE index and forks
throwaway auto-titled sessions, so the bot appears to have "lost its memory".

The fix: an *accidental* end reason (ws_orphan_reap) hitting a canonical Bot
Chat row skips only the DB end-write; explicit user boundaries still end it.
"""

from unittest.mock import MagicMock, patch

from tui_gateway.server import (
    _finalize_session,
    _is_canonical_bot_chat_row,
)


def _make_session(session_id="sess_1"):
    agent = MagicMock()
    agent.session_id = session_id
    return {
        "agent": agent,
        "history": [{"role": "user", "content": "x"}],
        "history_lock": None,
        "session_key": session_id,
    }


def _canonical_row(**overrides):
    row = {"id": "sess_1", "source": "desktop", "profile_name": "default",
           "title": "Bot Chat"}
    row.update(overrides)
    return row


class TestIsCanonicalBotChatRow:
    def test_matches_profile_scoped_title(self):
        assert _is_canonical_bot_chat_row(_canonical_row()) is True

    def test_requires_a_profile(self):
        assert _is_canonical_bot_chat_row(_canonical_row(profile_name="")) is False
        assert _is_canonical_bot_chat_row(_canonical_row(profile_name=None)) is False

    def test_title_match_is_exact(self):
        assert _is_canonical_bot_chat_row(_canonical_row(title="Bot Chat #2")) is False
        assert _is_canonical_bot_chat_row(_canonical_row(title="bot chat")) is False

    def test_none_and_empty_rows_are_not(self):
        assert _is_canonical_bot_chat_row(None) is False
        assert _is_canonical_bot_chat_row({}) is False


class TestFinalizeSkipsAccidentalReapOfCanonicalBotChat:
    @patch("tui_gateway.server._get_db")
    def test_ws_orphan_reap_does_not_end_the_canonical_row(self, mock_get_db):
        db = MagicMock()
        db.get_session.return_value = _canonical_row()
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="ws_orphan_reap")

        db.end_session.assert_not_called()

    @patch("tui_gateway.server._get_db")
    def test_explicit_user_close_still_ends_it(self, mock_get_db):
        """tui_close / session.close are real boundaries — keep ending the
        row so a deliberately closed Bot Chat doesn't haunt /resume."""
        db = MagicMock()
        db.get_session.return_value = _canonical_row()
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="tui_close")

        db.end_session.assert_called_once_with("sess_1", "tui_close")

    @patch("tui_gateway.server._get_db")
    def test_reap_of_an_ordinary_desktop_session_still_ends_it(self, mock_get_db):
        """Only the canonical title is protected; every other desktop session
        keeps the pre-existing reap behavior."""
        db = MagicMock()
        db.get_session.return_value = _canonical_row(title="Tell me about yourself")
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="ws_orphan_reap")

        db.end_session.assert_called_once_with("sess_1", "ws_orphan_reap")

    @patch("tui_gateway.server._get_db")
    def test_gateway_owned_sessions_keep_their_own_guard(self, mock_get_db):
        """The #60609 gateway-owner guard must keep working for Bot rows that
        ride a gateway platform source."""
        db = MagicMock()
        db.get_session.return_value = _canonical_row(source="telegram")
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="ws_orphan_reap")

        db.end_session.assert_not_called()

    @patch("tui_gateway.server._get_db")
    def test_missing_row_still_ended_on_reap(self, mock_get_db):
        """No state.db row → can't be a canonical chat — keep the legacy
        behavior."""
        db = MagicMock()
        db.get_session.return_value = None
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="ws_orphan_reap")

        db.end_session.assert_called_once_with("sess_1", "ws_orphan_reap")
