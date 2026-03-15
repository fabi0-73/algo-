"""
Telegram Notifier for Live AMD Signals

Sends formatted signal messages to a Telegram chat via the Bot API.
"""
import html
import logging

import requests

from src.live.signals import LiveSignal

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
DEFAULT_TIMEOUT = 10
RETRY_TIMEOUT = 15


class TelegramNotifier:
    """
    Sends live AMD signals to a Telegram chat using the Bot API.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self._session = requests.Session()

    def send_signal(self, signal: LiveSignal) -> bool:
        """
        Format the signal and send it to Telegram.
        Returns True if sent successfully, False otherwise.
        """
        text = self._format_signal(signal)
        return self._send_message(text)

    def send_test_message(self) -> bool:
        """
        Send a simple test message to verify bot token and chat_id.
        Returns True if sent successfully, False otherwise.
        """
        text = (
            "<b>AMD Live Scanner</b>\n\n"
            "Telegram connection OK. You will receive signals here when they fire."
        )
        return self._send_message(text)

    def _format_signal(self, signal: LiveSignal) -> str:
        """Build HTML message body for the signal."""
        ts = signal.timestamp
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "—"

        # Confluence tags: BOS, VOL, FVG, OB, etc.
        tags = []
        if signal.bos_confirmed:
            tags.append("BOS")
        if signal.volume_confirmed:
            tags.append("VOL")
        if signal.fvg_confluence:
            tags.append("FVG")
        if signal.ob_confluence:
            tags.append("OB")
        if signal.midnight_price_swept:
            tags.append("MID")
        confluence_tags = "  ".join(tags) if tags else "—"

        tier_label = signal.confidence.upper()
        risk_pct_str = f"{signal.risk_pct * 100:.1f}%" if signal.risk_pct else ""
        header = f"{signal.direction} {self._esc(signal.symbol)}  |  {tier_label} tier  ({risk_pct_str} risk)"
        block1 = (
            f"Entry:  {signal.entry_price:,.2f}  (LIMIT)\n"
            f"SL:     {signal.stop_loss:,.2f}\n"
            f"TP:     {signal.take_profit:,.2f}\n"
            f"R:R:    {signal.risk_reward:.2f}\n"
            f"Size:   {signal.position_size_lots:.2f} lots"
        )
        block2 = (
            f"Confluence: {signal.confluence_score}  |  {confluence_tags}\n"
            f"Judas Quality: {signal.judas_quality}\n"
            f"Range: {signal.consolidation_low:,.2f} - {signal.consolidation_high:,.2f}"
        )
        footer = ts_str

        return f"<b>{header}</b>\n\n<pre>{block1}</pre>\n\n{block2}\n\n<code>{footer}</code>"

    @staticmethod
    def _esc(s: str) -> str:
        """Escape for Telegram HTML: <, >, &."""
        return html.escape(str(s), quote=False)

    def _send_message(self, text: str) -> bool:
        """POST to Telegram Bot API with one retry on timeout."""
        url = TELEGRAM_API_BASE.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = self._session.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            if not r.ok:
                try:
                    err = r.json()
                    msg = err.get("description", r.text or str(r.status_code))
                except Exception:
                    msg = r.text or str(r.status_code)
                logger.error("Telegram API error %s: %s", r.status_code, msg)
                return False
            logger.info("Telegram: signal sent successfully")
            return True
        except requests.Timeout:
            logger.warning("Telegram: request timeout, retrying once...")
            try:
                r = self._session.post(url, json=payload, timeout=RETRY_TIMEOUT)
                if not r.ok:
                    try:
                        err = r.json()
                        msg = err.get("description", r.text or str(r.status_code))
                    except Exception:
                        msg = r.text or str(r.status_code)
                    logger.error("Telegram API error %s: %s", r.status_code, msg)
                    return False
                logger.info("Telegram: signal sent on retry")
                return True
            except Exception as e:
                logger.error("Telegram: send failed on retry: %s", e)
                return False
        except requests.RequestException as e:
            logger.error("Telegram: send failed: %s", e)
            return False
