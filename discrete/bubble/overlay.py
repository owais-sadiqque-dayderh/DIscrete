"""Bubble's status pill: a tiny AppKit process driven by JSON lines on stdin.

AppKit must own its process's main thread, and the daemon's main thread is
asyncio, so the pill lives in its own process (spawned once, kept warm, so a
trigger shows it in a frame). The panel is non-activating and ignores the
mouse: it never takes focus from the field the user is typing in.

Look: a dark HUD capsule under the caret. On the left, a conic-gradient ring
spins around a breathing core; the status line ("Searching your memory")
carries a light band that sweeps across the glyphs, the "thinking" shimmer
of agent chats such as Conductor; a monospaced timer counts on the right.
On success the ring settles into a solid green dot, then the capsule fades.

Protocol, one JSON object per line:
  {"op": "show", "anchor": [x, y, w, h] | null}   AX (top-left) coordinates
  {"op": "status", "text": "..."}
  {"op": "done", "text": "..."}
  {"op": "message", "text": "...", "tone": "info|warn|muted"}
  {"op": "hide"} | {"op": "quit"}
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time

from AppKit import (
    NSAnimationContext,
    NSApplication,
    NSAttributedString,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSFont,
    NSFontAttributeName,
    NSMakeRect,
    NSObject,
    NSPanel,
    NSScreen,
    NSTimer,
    NSView,
    NSVisualEffectView,
)
from PyObjCTools import AppHelper
from Quartz import (
    CABasicAnimation,
    CAGradientLayer,
    CALayer,
    CAMediaTimingFunction,
    CAShapeLayer,
    CATextLayer,
    CATransaction,
    CGPathCreateWithEllipseInRect,
    CGRectMake,
)

H = 34.0
PAD = 12.0
ORB = 14.0
TEXT_X = PAD + ORB + 9.0
TIMER_W = 42.0
FONT = 13.0
GAP_BELOW_CARET = 8.0

_ACCENT = (0.62, 0.53, 1.00)   # violet
_ACCENT2 = (0.36, 0.84, 1.00)  # cyan
_GREEN = (0.30, 0.85, 0.52)
_AMBER = (1.00, 0.72, 0.30)
_MUTED = (0.62, 0.64, 0.70)


def _cg(rgb, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], a).CGColor()


def _white(a):
    return NSColor.colorWithWhite_alpha_(1.0, a).CGColor()


def _font():
    try:
        return NSFont.systemFontOfSize_weight_(FONT, 0.23)  # medium
    except Exception:
        return NSFont.systemFontOfSize_(FONT)


def _timer_font():
    try:
        return NSFont.monospacedDigitSystemFontOfSize_weight_(11.0, 0.0)
    except Exception:
        return NSFont.systemFontOfSize_(11.0)


def _text_width(text: str, font) -> float:
    s = NSAttributedString.alloc().initWithString_attributes_(text, {NSFontAttributeName: font})
    return float(s.size().width)


def _text_layer(font, size, color, scale) -> CATextLayer:
    t = CATextLayer.layer()
    t.setFont_(font)
    t.setFontSize_(size)
    t.setForegroundColor_(color)
    t.setContentsScale_(scale)
    t.setAlignmentMode_("left")
    t.setTruncationMode_("end")
    return t


class _Ticker(NSObject):
    """NSTimer target. The pill itself is plain Python: PyObjC would turn every
    method of an NSObject subclass into an Objective-C selector."""

    def tick_(self, _timer):
        self.pill.tick()


class Pill:
    def __init__(self):
        self.started = time.monotonic()
        self.running = False
        self.fade_token = 0
        self.width = 220.0
        self._ticker = _Ticker.alloc().init()
        self._ticker.pill = self
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self):
        scale = float(NSScreen.mainScreen().backingScaleFactor()) if NSScreen.mainScreen() else 2.0
        style = 0 | (1 << 7)  # borderless | non-activating panel
        p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.width, H), style, NSBackingStoreBuffered, False)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(True)
        p.setLevel_(25)  # NSStatusWindowLevel: above normal and floating windows
        p.setIgnoresMouseEvents_(True)
        p.setHidesOnDeactivate_(False)
        p.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))  # all spaces | stationary | fullscreen aux
        p.setAlphaValue_(0.0)

        fx = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, self.width, H))
        fx.setMaterial_(13)       # HUD window
        fx.setBlendingMode_(0)    # behind window
        fx.setState_(1)           # always active
        fx.setWantsLayer_(True)
        fx.layer().setCornerRadius_(H / 2)
        fx.layer().setMasksToBounds_(True)
        fx.layer().setBorderWidth_(0.5)
        fx.layer().setBorderColor_(_white(0.14))
        fx.setAutoresizingMask_(2 | 16)

        host = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, self.width, H))
        host.setWantsLayer_(True)
        host.setAutoresizingMask_(2 | 16)
        fx.addSubview_(host)
        p.setContentView_(fx)
        root = host.layer()

        # Orb: conic ring (spins) + core (breathes).
        orb = CALayer.layer()
        orb.setFrame_(CGRectMake(PAD, (H - ORB) / 2, ORB, ORB))
        ring = CAGradientLayer.layer()
        ring.setFrame_(CGRectMake(0, 0, ORB, ORB))
        ring.setType_("conic")
        ring.setStartPoint_((0.5, 0.5))
        ring.setEndPoint_((0.5, 0.0))
        mask = CAShapeLayer.layer()
        mask.setFrame_(CGRectMake(0, 0, ORB, ORB))
        mask.setPath_(CGPathCreateWithEllipseInRect(CGRectMake(1.3, 1.3, ORB - 2.6, ORB - 2.6), None))
        mask.setLineWidth_(2.2)
        mask.setFillColor_(None)
        mask.setStrokeColor_(_white(1.0))
        mask.setLineCap_("round")
        ring.setMask_(mask)
        core = CALayer.layer()
        c = 5.0
        core.setFrame_(CGRectMake((ORB - c) / 2, (ORB - c) / 2, c, c))
        core.setCornerRadius_(c / 2)
        orb.addSublayer_(ring)
        orb.addSublayer_(core)
        root.addSublayer_(orb)

        # Status text: dim base + a shimmer band masked by the same glyphs.
        font = _font()
        base = _text_layer(font, FONT, _white(0.50), scale)
        shimmer = CAGradientLayer.layer()
        shimmer.setStartPoint_((0.0, 0.5))
        shimmer.setEndPoint_((1.0, 0.5))
        shimmer.setColors_([_white(0.0), _white(1.0), _white(0.0)])
        shimmer.setLocations_([0.0, 0.12, 0.24])
        glyphs = _text_layer(font, FONT, _white(1.0), scale)
        shimmer.setMask_(glyphs)
        root.addSublayer_(base)
        root.addSublayer_(shimmer)

        timer = _text_layer(_timer_font(), 11.0, _white(0.38), scale)
        timer.setAlignmentMode_("right")
        root.addSublayer_(timer)

        self.panel, self.fx, self.root = p, fx, root
        self.ring, self.ring_mask, self.core = ring, mask, core
        self.base, self.shimmer, self.glyphs, self.timer = base, shimmer, glyphs, timer
        self.font = font
        self.tick_timer = None
        self._layout(self.width)

    def _layout(self, width):
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        tw = width - TEXT_X - TIMER_W - PAD
        ty = (H - FONT * 1.25) / 2 - 1
        frame = CGRectMake(TEXT_X, ty, tw, FONT * 1.3)
        self.base.setFrame_(frame)
        self.shimmer.setFrame_(frame)
        self.glyphs.setFrame_(CGRectMake(0, 0, tw, FONT * 1.3))
        self.timer.setFrame_(CGRectMake(width - PAD - TIMER_W, (H - 13) / 2 - 1, TIMER_W, 14))
        CATransaction.commit()

    # -------------------------------------------------------------- animation

    def _spin(self, on: bool, tone=None):
        self.ring.removeAllAnimations()
        self.core.removeAllAnimations()
        if on:
            self.ring.setColors_([_cg(_ACCENT, 0.0), _cg(_ACCENT, 0.9), _cg(_ACCENT2, 1.0)])
            self.core.setBackgroundColor_(_cg(_ACCENT2, 1.0))
            rot = CABasicAnimation.animationWithKeyPath_("transform.rotation.z")
            rot.setFromValue_(0.0)
            rot.setToValue_(-2 * math.pi)
            rot.setDuration_(0.85)
            rot.setRepeatCount_(1e9)
            self.ring.addAnimation_forKey_(rot, "spin")
            breathe = CABasicAnimation.animationWithKeyPath_("transform.scale")
            breathe.setFromValue_(0.7)
            breathe.setToValue_(1.2)
            breathe.setDuration_(0.75)
            breathe.setAutoreverses_(True)
            breathe.setRepeatCount_(1e9)
            breathe.setTimingFunction_(CAMediaTimingFunction.functionWithName_("easeInEaseOut"))
            self.core.addAnimation_forKey_(breathe, "breathe")
        else:
            col = tone or _GREEN
            self.ring.setColors_([_cg(col, 1.0), _cg(col, 1.0), _cg(col, 1.0)])
            self.core.setBackgroundColor_(_cg(col, 1.0))
            pop = CABasicAnimation.animationWithKeyPath_("transform.scale")
            pop.setFromValue_(0.6)
            pop.setToValue_(1.0)
            pop.setDuration_(0.22)
            self.core.addAnimation_forKey_(pop, "pop")

    def _shimmer(self, on: bool):
        self.shimmer.removeAllAnimations()
        self.shimmer.setHidden_(not on)
        if on:
            a = CABasicAnimation.animationWithKeyPath_("locations")
            a.setFromValue_([-0.36, -0.24, -0.12])
            a.setToValue_([1.12, 1.24, 1.36])
            a.setDuration_(1.35)
            a.setRepeatCount_(1e9)
            a.setTimingFunction_(CAMediaTimingFunction.functionWithName_("easeInEaseOut"))
            self.shimmer.addAnimation_forKey_(a, "sweep")

    def _set_text(self, text: str, *, bright: bool):
        need = TEXT_X + _text_width(text, self.font) + 10 + TIMER_W + PAD
        width = max(180.0, min(520.0, need))
        if abs(width - self.width) > 1:
            self.width = width
            f = self.panel.frame()
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.18)
            self.panel.animator().setFrame_display_(NSMakeRect(f.origin.x, f.origin.y, width, H), True)
            NSAnimationContext.endGrouping()
            self._layout(width)
        self.base.setString_(text)  # implicit crossfade of the glyphs
        self.glyphs.setString_(text)
        self.base.setForegroundColor_(_white(0.92 if bright else 0.50))

    # ---------------------------------------------------------------- commands

    def handle_(self, msg):
        op = msg.get("op")
        if op == "show":
            self._show(msg.get("anchor"))
        elif op == "status":
            self.running = True
            self._spin(True)
            self._shimmer(True)
            self._set_text(str(msg.get("text") or "Thinking"), bright=False)
        elif op == "done":
            self._finish(str(msg.get("text") or "Done"), _GREEN, hold=0.9)
        elif op == "message":
            tone = {"warn": _AMBER, "muted": _MUTED}.get(msg.get("tone"), _ACCENT2)
            self._finish(str(msg.get("text") or ""), tone, hold=2.6)
        elif op == "hide":
            self._fade(0.0)
        elif op == "quit":
            AppHelper.stopEventLoop()

    def _show(self, anchor):
        self.fade_token += 1
        self.started = time.monotonic()
        self.running = True
        self._spin(True)
        self._shimmer(True)
        self._set_text("Thinking", bright=False)
        self.timer.setString_("")
        x, y = self._place(anchor)
        self.panel.setFrame_display_(NSMakeRect(x, y - 6, self.width, H), True)
        self.panel.orderFrontRegardless()
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.16)
        self.panel.animator().setAlphaValue_(1.0)
        self.panel.animator().setFrame_display_(NSMakeRect(x, y, self.width, H), True)
        NSAnimationContext.endGrouping()
        if self.tick_timer is None:
            self.tick_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.1, self._ticker, "tick:", None, True)

    def _place(self, anchor):
        screens = list(NSScreen.screens() or [])
        top = float(screens[0].frame().size.height) if screens else 900.0
        if anchor:
            ax, ay, aw, ah = (float(v) for v in anchor)
            px, below = ax, top - (ay + ah) - GAP_BELOW_CARET - H
            above = top - ay + GAP_BELOW_CARET
        else:
            m = NSEvent.mouseLocation()
            px, below, above = m.x + 14, m.y - H - 18, m.y + 18
        scr = next((s for s in screens if _contains(s.frame(), px, below + H)), screens[0] if screens else None)
        if scr is None:
            return px, below
        vf = scr.visibleFrame()
        y = below if below >= vf.origin.y + 4 else above
        x = min(max(px, vf.origin.x + 6), vf.origin.x + vf.size.width - self.width - 6)
        y = min(max(y, vf.origin.y + 4), vf.origin.y + vf.size.height - H - 4)
        return x, y

    def _finish(self, text, tone, *, hold):
        self.running = False
        self._spin(False, tone)
        self._shimmer(False)
        self._set_text(text, bright=True)
        if self.panel.alphaValue() < 0.5:  # message without a prior show
            self._show(None)
            self.running = False
            self._spin(False, tone)
            self._shimmer(False)
            self._set_text(text, bright=True)
        self.fade_token += 1
        token = self.fade_token
        AppHelper.callLater(hold, self._fade_if, token)

    def _fade_if(self, token):
        if token == self.fade_token:
            self._fade(0.0)

    def _fade(self, _):
        self.running = False
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.28)
        self.panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()

    def tick(self):
        if not self.running:
            return
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self.timer.setString_(f"{time.monotonic() - self.started:.1f}s")
        CATransaction.commit()


def _contains(frame, x, y) -> bool:
    return (frame.origin.x <= x <= frame.origin.x + frame.size.width
            and frame.origin.y <= y <= frame.origin.y + frame.size.height)


def _reader(pill: Pill) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        AppHelper.callAfter(pill.handle_, msg)
    AppHelper.callAfter(AppHelper.stopEventLoop)


def main() -> None:
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(2)  # prohibited: no Dock icon, never becomes active
    pill = Pill()
    threading.Thread(target=_reader, args=(pill,), daemon=True).start()
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
