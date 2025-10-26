# logs

Default output directory for log files produced by `my_library.utils.logger`. Files are rotated automatically to prevent uncontrolled growth.

## Notes

- Logs are only written when `Logger(..., save_to_file=True)` or the environment variable `SAVE_LOG_TO_FILE` is set to `True`.
- Each run creates a timestamped file, e.g., `CustomLightGBM_20240101_120000.log`.
- Log files are safe to delete between experiments; no code depends on them at runtime.
