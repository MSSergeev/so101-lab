"""Performance optimization utilities for Isaac Sim/Lab."""


def disable_rate_limiting():
    """Disable Isaac Sim rate limiting for maximum simulation speed.

    This removes the artificial FPS cap that Isaac Sim applies by default.
    Should be called after AppLauncher but before creating environments.

    See docs/PERFORMANCE_TUNING.md for details.
    """
    try:
        import carb.settings

        settings = carb.settings.get_settings()

        # Disable rate limiting on main and rendering loops
        settings.set("/app/runLoops/main/rateLimitEnabled", False)
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", False)

        # Additional throttling settings
        settings.set("/app/renderer/skipWhileMinimized", False)
        settings.set("/app/renderer/sleepMsOnFocus", 0)
        settings.set("/app/renderer/sleepMsOutOfFocus", 0)

        print("[PERF] Rate limiting disabled")
    except Exception as e:
        print(f"[PERF] Failed to disable rate limiting: {e}")
