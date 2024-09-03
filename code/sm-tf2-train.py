import os
import json
import logging
import time
import tensorflow as tf
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["GRPC_FAIL_FAST"] = "use_caller"

model_dir = os.environ.get('SM_CHANNEL_MODEL_OUTPUT', '/tmp')
training_data = os.environ['SM_CHANNEL_TRAINING']
batch_size = 64
total_samples = 64*100*2

sm_training_env = os.environ.get('SM_TRAINING_ENV')
training_env = json.loads(sm_training_env)
job_name = training_env.get('job_name')

done_file_name = f"{job_name}_DONE"


def check_done_file(model_dir):
    done_file_path = os.path.join(model_dir, done_file_name)
    file_exists = os.path.exists(done_file_path)
    return file_exists


def create_done_file(model_dir):
    done_file_path = os.path.join(model_dir, done_file_name)
    with open(done_file_path, 'w') as f:
        f.write('Training completed')
    return done_file_path


def parse_tfrecord(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([10 * 10], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.reshape(parsed_features['x'], [10, 10])
    y = tf.reshape(parsed_features['y'], [1])
    return x, y


def replace_master_with_chief():
    if 'TF_CONFIG' not in os.environ:
        raise ValueError("TF_CONFIG not found in environment variables")

    tf_config = json.loads(os.environ['TF_CONFIG'])
    if 'cluster' in tf_config and 'master' in tf_config['cluster']:
        tf_config['cluster']['chief'] = tf_config['cluster']['master']
        del tf_config['cluster']['master']

    if 'task' in tf_config and tf_config['task']['type'] == 'master':
        tf_config['task']['type'] = 'chief'

    os.environ['TF_CONFIG'] = json.dumps(tf_config)


def get_dataset():
    global total_samples

    tfrecord_files = [os.path.join(training_data, f) for f in os.listdir(training_data) if f.endswith('.tfrecord')]
    logger.info(f"Found {len(tfrecord_files)} TFRecord files: {tfrecord_files}")

    train_dataset = tf.data.TFRecordDataset(tfrecord_files).map(parse_tfrecord)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset


def train():
    replace_master_with_chief()

    tf_config = json.loads(os.environ['TF_CONFIG'])
    logger.info(f"TF_CONFIG: {tf_config}")

    cluster = tf_config['cluster']
    task = tf_config['task']

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        tf.train.ClusterSpec(cluster),
        task_type=task['type'],
        task_id=task['index'],
        num_accelerators={'GPU': 0},
        rpc_layer='grpc'
    )

    # we will only setup worker only, because SageMaker will init a ps in each instance.
    if cluster_resolver.task_type == 'worker':
        print(f"[{time.time()}] Start worker({cluster_resolver.task_id})...")

        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True)

        # keep the worker alive until we receive a shutdown signal
        while not check_done_file(model_dir):
            time.sleep(10)
        logger.info(f"Worker {cluster_resolver.task_id} detected completion, shutting down.")

        return

    if cluster_resolver.task_type != 'chief':
        return

    variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=len(cluster.get('ps', [])))

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner
    )

    global_batch_size = batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(10, 10)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=['mae', 'mse']
        )

    @tf.function
    def train_step(iterator):
        def step_fn(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                per_example_loss = model.loss(y, predictions)
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update metrics
            model.compiled_metrics.update_state(y, predictions)

            return loss

        return strategy.run(step_fn, args=(next(iterator),))

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

    @tf.function
    def per_worker_dataset_fn():
        return get_dataset()

    distributed_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    distributed_iterator = iter(distributed_dataset)

    # Calculate steps_per_epoch
    steps_per_epoch = tf.math.ceil(total_samples / global_batch_size).numpy()
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    num_epochs = 5
    steps_per_epoch = 10

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for step in range(steps_per_epoch):
            try:
                loss = coordinator.schedule(train_step, args=(distributed_iterator,))
                loss = loss.fetch()
                if step % 10 == 0:
                    # Get current metric values
                    metric_results = {m.name: m.result().numpy() for m in model.metrics}
                    logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss}, Metrics: {metric_results}")
            except tf.errors.OutOfRangeError:
                logger.info(f"Reached end of dataset at step {step}")
                break

        # Reset metrics at the end of each epoch
        model.reset_metrics()

    coordinator.join()
    logger.info("Training completed")

    done_file = create_done_file(model_dir)
    logger.info(f"Created DONE file: {done_file}. Workers should start shutting down.")

    # Create model output folder
    full_path = os.path.join(model_dir, job_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Save the model
    model.save(full_path)
    logger.info("Model saved")

    # Copy to /opt/ml/model, SageMaker will sync to S3
    shutil.copytree(full_path, "/opt/ml/model/", dirs_exist_ok=True)
    logger.info("Model copied to /opt/ml/model/")


if __name__ == "__main__":
    train()