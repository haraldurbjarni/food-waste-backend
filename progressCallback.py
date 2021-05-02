import keras

class ProgressCallback(keras.callbacks.Callback):
    """Callback that prints metrics to stdout.
    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
    # Raises
        ValueError: In case of invalid `count_mode`.
    """
    verbose = None
    epochs = None
    log_values = None
    target = 0
    seen = 0
    progbar = None

    progress_callback = None

    def __init__(self, progressCallback, count_mode='samples'):
        super(ProgressCallback, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        
        self.progress_callback = progressCallback
        progressCallback(0)

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress_callback(epoch / self.epochs)

