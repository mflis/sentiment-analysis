import io

import matplotlib.pyplot as plt
import tensorflow as tf

from .FilterTensorBoard import FilterTensorBoard


class ImagesTensorBoard(FilterTensorBoard):
    """
   plot images to tensorboard
    """

    def __init__(self, plotting_functions, suffix, *args, **kwargs):
        self.plotting_functions = plotting_functions
        self.suffix = suffix
        super(ImagesTensorBoard, self).__init__(*args, **kwargs)

    def evaluate(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_pred_float = self.model.predict(x_val, verbose=0)
        # todo consider adjustable binarization threshold
        return y_val, y_pred_float

    def on_epoch_end(self, epoch, logs=None):
        super(ImagesTensorBoard, self).on_epoch_end(epoch, logs)

        for name, plot_func in self.plotting_functions.items():
            self.write_metric(name, plot_func, epoch)

    def write_metric(self, name, plot_func, epoch):
        # Prepare the plot
        buf = io.BytesIO()
        y_val, y_pred_float = self.evaluate()
        plot_func(y_val, y_pred_float, buf)
        plt.close('all')
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # todo maybe write more efficiently
        summary_op = tf.summary.image("image", image, family='{}-{}'.format(name, self.suffix))  # Session
        with tf.Session() as sess:
            # Run
            summary = sess.run(summary_op)
            # Write summary
            self.writer.add_summary(summary, global_step=epoch)
            self.writer.flush()
