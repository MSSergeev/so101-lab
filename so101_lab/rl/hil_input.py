# Status: incomplete — real robot HIL-SERL (Phase 5.7) not yet implemented
"""HIL toggle: keyboard + viewer commands for human-in-the-loop takeover."""

from so101_lab.utils.shm_preview import read_command


class HILToggle:
    """Toggle for human-in-the-loop takeover.

    Supports two input sources (can work in parallel):
    - GUI mode: Enter key via carb.input, X key for episode reset
    - Preview mode: Space/X commands from camera_viewer.py via shared memory
    """

    def __init__(self, gui: bool = True):
        self._active = False
        self._pending_reset = False
        self._sub = None

        if gui:
            import carb
            import omni

            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_key
            )
            print("[HIL] Press ENTER in Isaac Sim window to toggle takeover")
            print("[HIL] Press X in Isaac Sim window to reset episode")

    @property
    def is_active(self) -> bool:
        return self._active

    def poll_commands(self) -> str | None:
        """Poll viewer commands via shared memory.

        Returns "reset" if episode reset requested, None otherwise.
        Toggle is handled internally.
        """
        # Check GUI pending reset first
        if self._pending_reset:
            self._pending_reset = False
            return "reset"

        cmd = read_command()
        if cmd == "teleop":
            self._active = not self._active
            print(f"[HIL] TAKEOVER {'ON' if self._active else 'OFF'}")
        elif cmd == "discard":
            return "reset"
        return None

    def stop(self):
        if self._sub is not None:
            import carb

            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _on_key(self, event, *args, **kwargs):
        import carb

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "ENTER":
                self._active = not self._active
                print(f"[HIL] TAKEOVER {'ON' if self._active else 'OFF'}")
            elif event.input.name == "X":
                self._pending_reset = True
        return True
