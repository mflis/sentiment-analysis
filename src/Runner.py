import tensorflow as tf


class Runner(object):
    def run(self):
        # with graph.as_default():
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
            test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
            iteration = 1
            self.buildGraph()
            inputObj = Input(self.config)
            inputs, embeddings, labels = inputObj.readInput()

            train_x, val_x, test_x = split(inputs)
            train_y, val_y, test_y = split(labels)

            for e in range(epochs):
                state = sess.run(initial_state)

                for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 0.5,
                            initial_state: state}
                    summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict=feed)
                    #             loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                    train_writer.add_summary(summary, iteration)

                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for x, y in get_batches(val_x, val_y, batch_size):
                            feed = {inputs_: x,
                                    labels_: y[:, None],
                                    keep_prob: 1,
                                    initial_state: val_state}
                            #                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                            summary, batch_acc, val_state = sess.run([merged, accuracy, final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration += 1
                    test_writer.add_summary(summary, iteration)
                    saver.save(sess, "checkpoints/sentiment_manish.ckpt")
            saver.save(sess, "checkpoints/sentiment_manish.ckpt")
