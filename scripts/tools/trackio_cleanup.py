#!/usr/bin/env python3
"""Interactive cleanup of trackio experiment runs.

Navigate: projects → runs → delete selected.

Usage:
    python scripts/tools/trackio_cleanup.py
"""

from trackio.sqlite_storage import SQLiteStorage


def choose(prompt: str, options: list[str], allow_multi: bool = False) -> list[str] | str | None:
    """Interactive selector. Returns chosen item(s) or None for back/quit."""
    if not options:
        print("  (empty)")
        return None

    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_multi:
        print(f"  a. Select all")
    print(f"  q. {'Back' if allow_multi else 'Quit'}")

    while True:
        raw = input(f"\n{prompt}: ").strip().lower()
        if raw == "q":
            return None
        if allow_multi and raw == "a":
            return list(options)
        # Parse comma-separated indices for multi-select
        if allow_multi and "," in raw:
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                if all(1 <= i <= len(options) for i in indices):
                    return [options[i - 1] for i in indices]
            except ValueError:
                pass
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return [options[idx - 1]] if allow_multi else options[idx - 1]
        except ValueError:
            pass
        print("  Invalid input, try again")


def show_run_info(project: str, run: str):
    """Print run summary."""
    config = SQLiteStorage.get_run_config(project, run)
    metrics = SQLiteStorage.get_all_metrics_for_run(project, run)
    print(f"\n  Run: {run}")
    print(f"  Metrics: {', '.join(metrics[:10])}" + ("..." if len(metrics) > 10 else ""))
    if config:
        steps = config.get("num_steps", "?")
        mode = "bc-pretrain" if config.get("bc_pretrain") else \
               "offline" if config.get("no_online") else \
               "online-only" if config.get("no_offline") else "online+offline"
        if config.get("hil"):
            mode += "+hil"
        print(f"  Mode: {mode}, steps: {steps}")


def main():
    print("=== Trackio Cleanup ===\n")

    while True:
        projects = SQLiteStorage.get_projects()
        if not projects:
            print("No projects found.")
            return

        print("\nProjects:")
        project = choose("Select project (q=quit)", projects)
        if project is None:
            return

        while True:
            runs = SQLiteStorage.get_runs(project)
            if not runs:
                print(f"\n  No runs in '{project}'")
                break

            print(f"\nRuns in '{project}':")
            for r in runs:
                config = SQLiteStorage.get_run_config(project, r)
                tag = ""
                if config:
                    if config.get("bc_pretrain"):
                        tag = " [BC]"
                    elif config.get("no_online"):
                        tag = " [offline]"
                    elif config.get("hil"):
                        tag = " [HIL]"
                tag_str = tag
                print(f"  - {r}{tag_str}")

            selected = choose("Select runs to delete (comma-separated, a=all, q=back)", runs, allow_multi=True)
            if selected is None:
                break

            # Show info and confirm
            for r in selected:
                show_run_info(project, r)

            confirm = input(f"\n  Delete {len(selected)} run(s)? [y/N]: ").strip().lower()
            if confirm == "y":
                for r in selected:
                    if SQLiteStorage.delete_run(project, r):
                        print(f"  Deleted: {r}")
                    else:
                        print(f"  Failed: {r}")

                # If project is now empty, offer to delete it
                remaining = SQLiteStorage.get_runs(project)
                if not remaining:
                    rm = input(f"\n  Project '{project}' is empty. Delete project? [y/N]: ").strip().lower()
                    if rm == "y":
                        from trackio import delete_project
                        delete_project(project, force=True)
                        print(f"  Project deleted.")
                        break
            else:
                print("  Cancelled.")


if __name__ == "__main__":
    main()
