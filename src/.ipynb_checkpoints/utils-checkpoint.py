def is_unavailable_error(error: Exception) -> bool:
    """Detect a 503 "model overloaded / high demand" style error.

    The google-genai client does not expose a single stable error type
    across SDK versions, so this checks both the ``code``/``status``
    attributes (when present) and the string representation of the
    error for the usual 503 markers.
    """

    status_code = getattr(error, "code", None) or getattr(
        error, "status_code", None
    )

    if status_code == 503:
        return True

    error_text = str(error).upper()

    return (
        "503" in error_text
        or "UNAVAILABLE" in error_text
        or "OVERLOADED" in error_text
        or "HIGH DEMAND" in error_text
    )


def call_with_retry(
    function: Callable[[], Any],
    max_retries: int = MAX_RETRIES,
    max_retries_unavailable: int = MAX_RETRIES_UNAVAILABLE,
):
    """Run an API call with exponential retry delays.

    A 503 "high demand"/"UNAVAILABLE" error is treated separately: it
    reflects server-side overload rather than a transient client-side
    hiccup, so it is retried at most ``max_retries_unavailable`` times
    instead of the usual ``max_retries``, to avoid burning through the
    full backoff schedule (up to 8s per attempt) on errors that rarely
    resolve within a single run.
    """

    last_error = None
    attempt = 0

    while True:
        attempt += 1

        try:
            return function()

        except Exception as error:
            last_error = error

            unavailable = is_unavailable_error(error)
            effective_max = (
                max_retries_unavailable
                if unavailable
                else max_retries
            )

            if attempt >= effective_max:
                break

            wait_seconds = 2 ** attempt

            reason = (
                "503/high demand"
                if unavailable
                else type(error).__name__
            )

            print(
                f"Request failed ({reason}). Retrying in "
                f"{wait_seconds} seconds. Error: {error}"
            )

            time.sleep(wait_seconds)

    raise last_error