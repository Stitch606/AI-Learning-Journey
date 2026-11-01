"""
ZERO GPU HELPER.

On a Hugging Face ZeroGPU Space, a function decorated with @spaces.GPU gets a GPU
only while it runs. Everywhere else (Colab, your computer, a normal Space) the
decorator does nothing, so the same code runs in all places.

This file imports nothing from the project, and contains no AI.

Use it on the functions of YOUR code that need the GPU:

    from gpu import gpu

    @gpu
    def detect(frame):            # gets a GPU while it runs
        ...

    @gpu(duration=120)            # allow up to 120 seconds of GPU per call
    def heavy_job(video):
        ...
"""
try:
    import spaces                 # installed by Hugging Face on ZeroGPU Spaces
except ImportError:               # Colab / local: not installed, decorator becomes a no-op
    spaces = None


def gpu(fn=None, **options):
    """Works as @gpu and as @gpu(duration=...)."""
    if spaces is None:
        return fn if fn is not None else (lambda f: f)
    if fn is not None:
        return spaces.GPU(fn)
    return spaces.GPU(**options)


@gpu
def gpu_ready():
    """Does nothing. ZeroGPU refuses to start a Space that has no @spaces.GPU function
    ('No @spaces.GPU function detected during startup'); this placeholder satisfies that check.
    Once your own code has a real @gpu function, you may delete this one."""
    return True
