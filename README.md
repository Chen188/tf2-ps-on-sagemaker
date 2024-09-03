## 背景
Tensorflow（TF） 作为一款优秀的机器学习框架，自2017年发布1.0版本以来，便引来了大量用户的关注并在自己的生产环境中进行使用。随着 Tensorflow 的不断演进，其于2年后便推出更加高效的 2.0 版本，目前2个分支最新的版本分别为 1.15.4(2020年9月) 及 2.16.2（2024年6月）。由于 1.x 版本已经多年未更新，且 TF2 中引入了许多有价值的更新，例如：更紧密的Keras的集成、改进的 TF Datasets 提高了性能、可扩展性及易用性、支持 Keras Model.fit 等，一些客户便萌生了将原有 1.x 版本训练代码迁移到 2.x 的想法。

本文将总结分享客户从TF 1.15 迁移升级到 TF 2.14 过程中遇到的一些问题及应对方法，并以一个简单的例子给出基于 TF 2 的分布式训练代码。

## TF 2 和 TF 1 中 PS 训练差异
在实际的客户项目中，通常数据量非常大或者模型参数量较大，需要使用到分布式训练策略。TF 2 中支持多种分布式训练策略，例如：MirroredStrategy、MultiWorkerMirroredStrategy、ParameterServerStrategy 等。Parameter Server Strategy（PSS） 作为一个支持异步的参数更新的数据并行策略，提供了非常好的分布式训练性能。

TF 1 通过 Estimator 训练模型的代码中内置了自动初始化 Parameter Server（PS） 分布式集群的逻辑，具体见代码片段，可以实现从环境变量 TF_CONFIG 中自动初始化集群的信息。因此，在 TF 1 中可以通过配置 TF_CONFIG 环境变量以及手动设置集群信息两种集群初始化方法。

TF 2 修改了 PSS 的初始化策略，需要用户主动创建策略(Strategy)，然后通过 with strategy.scope() 向 Keras API 或者自定义训练环境传递分布式策略信息。否则，无法实现分布式的训练效果。此外，TF 2 简化了节点的类型，并建议通过一个中心协调器来为 WORKER、PS 节点创建资源、分发函数计算等，用户可以在训练脚本中根据节点的任务类型（task_type） 设置不同的工作内容，详见文档。还需要注意的是，使用 TF 2.14 结合 PS 多机分布式训练模型时，要使用一个共享文件存储系统来保存模型及checkpoints文件，否则会导致训练异常。

## Amazon SageMaker 对分布式训练的支持
Amazon SageMaker 作为一个端到端机器学习平台，提供了丰富的功能模块，包括数据预处理、模型训练/调优、模型部署/监控、MLOps 等。其中，在模型训练模块中，SageMaker 针对不同的训练框架提供了大量的预构建深度学习环境(DLC, Deep Learning Container)，用来支持不同的框架版本及分布式训练优化等。

以 TF 为例，SageMaker 内置了对 Parameter Server Strategy、MultiWorker Mirrored Strategy 及 SageMaker Distributed Data-parallel(SMDDP) 的支持。此外，SageMaker 也支持多种文件系统（如S3、EFS、FSx for Lustre）以及访问模式（如 File、FastFile、Pipe 模式等），用以满足不同数据规模、不同分布式训练方式下对数据读取的要求，具体可以参考文档。

SageMaker Tensorflow DLC 中与模型训练有关的 pip 库有三个：  
- SageMaker TensorFlow training toolkit：提供针对 Tensorflow 分布式训练的支持，除了本文中用到的 Parameter Server Strategy 外，还支持 MultiWorkerMirroredStrategy 及 SageMaker Distributed Data-parallel(SMDDP)。 
- SageMaker TensorFlow extensions：提供对 PipeModeDataset 的支持，用来以 Pipe 模式直接读取 S3 文件，而无需下载完整的文件内容到训练机器的本地存储。目前 SageMaker 已经支持 Fast File 模式，以文件系统方式访问 S3 中的文件，客户可以直接使用此模式而无需过多关注Pipe模式。
- SageMaker training toolkit：主要目的是执行客户设定的入口训练代码文件。

### SageMaker PS 原理解析
SageMaker TensorFlow training toolkit 中与 PS 分布式有关的代码片段如下：

- 初始化 PS 分布式训练的环境变量，并在每台实例上拉起对应的 PS 进程
  ```python
  def _build_tf_config_for_ps(hosts, current_host, ps_task=False):
      masters = hosts[:1]
      workers = hosts[1:]
      ps = hosts if len(hosts) > 1 else None
  
      def host_addresses(hosts, port=2222):
          return ["{}:{}".format(host, port) for host in hosts]
  
      tf_config = {"cluster": {"master": host_addresses(masters)}, "environment": "cloud"}
  
      if ps:
          tf_config["cluster"]["ps"] = host_addresses(ps, port="2223")
  
      if workers:
          tf_config["cluster"]["worker"] = host_addresses(workers)
  
      if ps_task:
          if ps is None:
              raise ValueError(
                  "Cannot have a ps task if there are no parameter servers in the cluster"
              )
          task_type = "ps"
          task_index = ps.index(current_host)
      elif _is_host_master(hosts, current_host):
          task_type = "master"
          task_index = 0
      else:
          task_type = "worker"
          task_index = workers.index(current_host)
  
      tf_config["task"] = {"index": task_index, "type": task_type}
      return tf_config
      
  ...
  
  def _run_ps(env, cluster):
      logger.info("Running distributed training job with parameter servers")
  
      cluster_spec = tf.train.ClusterSpec(cluster)
      task_index = env.hosts.index(env.current_host)
  
      no_gpu_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
  
      server = tf.distribute.Server(
          cluster_spec, job_name="ps", task_index=task_index, config=no_gpu_config
      )
  
      multiprocessing.Process(target=lambda: server.join()).start()
  ```
- 启动用户的训练脚本，同时为脚本传入 TF_CONFIG 环境变量
  ```
  def _run_worker(env, cmd_args, tf_config):
      env_vars = env.to_env_vars()
      env_vars["TF_CONFIG"] = json.dumps(tf_config)
  
      entry_point.run(
          uri=env.module_dir,
          user_entry_point=env.user_entry_point,
          args=cmd_args,
          env_vars=env_vars,
          capture_error=True,
      )
  ```

## 在 SageMaker 中执行 PS 分布式训练
### TF 1.15 的分布式训练代码示例
以下是通过环境变量配置集群的方法示例，此方法对用户来讲，几乎不需要调整训练代码：

```python
import tensorflow as tf
import os

# print(os.environ['TF_CONFIG'])
# >>> example output
# {"cluster": {"master": ["algo-1:2222"], "ps": ["algo-1:2223", "algo-2:2223"], "worker": ["algo-2:2222"]}, "environment": "cloud", "task": {"index": 0, "type": "master"}}

run_config=tf.estimator.RunConfig()

#>>> example output
# INFO:tensorflow:TF_CONFIG environment variable: {'cluster': {'master': ['algo-1:2222'], 'ps': ['algo-1:2223', 'algo-2:2223'], 'worker': ['algo-2:2222']}, 'environment': 'cloud', 'task': {'index': 0, 'type': 'master'}}

run_config.session_config
#>>> example output
# device_filters: "/job:ps"
# device_filters: "/job:master"
# allow_soft_placement: true
# graph_options {
#   rewrite_options {
#     meta_optimizer_iterations: ONE
#   }
# }

model = tf.estimator.Estimator(
    model_fn=model_fn,
    ...,
    config=run_config,
)

tf.estimator.train_and_evaluate(model , train_spec, eval_spec)
```

以下是手动初始化集群配置的代码示例，其中，我们可以根据模型训练机器的 CPU 数量调整intra_op_parallelism_threads、inter_op_parallelism_threads 等训练参数，用来提高模型训练效率 ：

```python
tf_config = json.loads(os.environ["TF_CONFIG"])
index = tf_config["task"]["index"]

device_filters = ["/job:ps"]
if str(tf_config["task"]["type"]) == "master":
    device_filters.append("/job:master")
else:
    worker_index = "/job:worker/task:" + str(index)
    device_filters.append(worker_index)

config = tf.ConfigProto(
    allow_soft_placement=True,
    device_count={"CPU": num_cpus},
    intra_op_parallelism_threads=num_cpus,
    inter_op_parallelism_threads=num_cpus,
    device_filters=device_filters,
)

run_config = tf.estimator.RunConfig().replace(session_config=config)

model = tf.estimator.Estimator(
    model_fn=model_fn,
    ...,
    config=run_config,
)

tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
```

代码示例可以参考 DeepFM on SageMaker。

### TF 2.14 的分布式训练代码示例
如“TF 2 和 TF 1 中 PS 训练差异”章节所示，在 TF 2.14 中，如果期望使用 PS 做分布式训练，则需要关注以下配置：

- TF_CONFIG 环境变量，因为 TF 2 中的节点类型发生了变化，因此，我们需要在初始化 TF 分布式训练策略之前调整 SageMaker DLC 预置的 TF_CONFIG 环境变量中的节点类型，否则，TF 会因为节点类型无效进而初始化失败。
- 共享文件存储，用来保存训练的 checkpoint 及模型文件。这里，我们可以通过挂载一个EFS 文件系统，并将 checkpoints 及模型结果保存在此文件系统即可。
- 在模型完成训练之后，可以考虑将模型文件复制一份到 /opt/ml/model 一份，这样的话，SageMaker 会自动将模型文件打包上传到 S3 一份，方便做后续的模型文件管理。在整个任务训练结束后，您可以看到类似的日志：
```2024-xx-xx 15:15:15 Uploading - Uploading generated training model```

接下来，我们将展示具体的代码逻辑：

1. TF_CONFIG 配置调整

    这里我们主要是为了修改 TensorFlow 分布式训练的配置，将 "master" 角色替换为 "chief" 角色
  
    ```python
    # update TF_CONFIG
    def replace_master_with_chief():
        tf_config = json.loads(os.environ['TF_CONFIG'])
        if 'master' in tf_config['cluster']:
            tf_config['cluster']['chief'] = tf_config['cluster']['master']
            del tf_config['cluster']['master']
            
        if tf_config['task']['type'] == 'master':
            tf_config['task']['type'] = 'chief'
    
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    ```
2. 模型训练

    ```python
    def train():
        logger.info("Starting train function")
        
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
    
        # we will only setup worker only, cause SageMaker will init a ps in each instance.
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
    
        # save model to EFS
        save_model()
    ```

    其中各个组件的功能如下：
  
    - cluster_resolver: 负责解析并提供分布式训练所需的集群配置信息，也就是根据 TF_CONFIG 解析集群配置。
    - variable_partitioner: 定义模型变量在 PS 之间的分片策略。
    - strategy: 负责协调模型参数在workers和 PS 之间的分布、同步和更新，优化计算资源的利用和训练效率。
    - coordinator: 协调分布式训练过程，管理数据集和训练步骤的执行。
    - worker 退出逻辑: 通过检测模型完成的信号文件，以实现worker节点的生命周期管理，确保其在训练结束后正确退出。

3. 模型保存

    ```python
    def save_model()
        # Create model output folder, model_dir points EFS
        full_path = os.path.join(model_dir, job_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
        # Save the model
        model.save(full_path)
        logger.info("Model saved")
    
        # Copy to /opt/ml/model, SageMaker will sync to S3
        shutil.copytree(full_path, "/opt/ml/model/", dirs_exist_ok=True)
        logger.info("Model copied to /opt/ml/model/")
    ```
  
    这里，我们将模型结果保存在EFS中，并复制一份到 /opt/ml/model/ 路径。

### 其他配置注意事项

#### 提交 SageMaker 模型训练作业

1. 定义共享存储

    注意：您需要提前创建文件系统，具体创建方法可以参考文档。需要注意的是，您的文件系统需要设置合适的安全组规则，以允许 SageMaker 访问。
    
    ```python
    from sagemaker.inputs import FileSystemInput
    
    efs_fs = sagemaker.inputs.FileSystemInput(
        file_system_id = 'fs-7dxxxx88', # 文件系统 id
        file_system_type='EFS', # EFS | FSxLustre
        directory_path='/', # EFS 文件系统中的路径
        file_system_access_mode='rw', # 默认值为只读('ro')
    )
    ```

2. 提交训练作业
    ```python
    import sagemaker
    from sagemaker.tensorflow.estimator import TensorFlow
    from datetime import datetime
    import os
    
    train_instance_type = 'ml.m5.4xlarge'
    train_instance_count = 2
    
    distributions = {'parameter_server': {'enabled': True}}
    
    base_job_name='tf2-ps'
    
    estimator = TensorFlow(
        entry_point='train.py',  # 模型训练的启动脚本
        ...,
        subnets = [
           'subnet-011da1xxxxxx41e1c' # 建议配置为私有子网，并与文件系统选择同一个可用区
        ],
        security_group_ids = [
           'sg-0cc9803xxxxx6eeb1', # SageMaker 训练实例上绑定的安全组，要确保可以访问EFS
        ],
    #    keep_alive_period_in_seconds=600,
    #    enable_remote_debug=True,
    )
    ```
  
    如果您期望后续的训练作业可以复用已经启动的计算资源，则可以配置 keep_alive_period_in_seconds 参数，最大为 3600s，对于需要进行多次迭代或微调的模型，可以快速启动下一轮训练，从而提高代码调试效率。
  
    如果您需要使用 SageMaker Remote Debugging 来调试代码，则可以将 enable_remote_debug 设置为 True，之后就可以通过 CLI 工具远程登陆到训练容器中，具体可以参考文档。

#### 网络及安全组配置
在提交模型训练作业的时候，您需要注意 subnets 参数，我们建议您：

- 如果您需要访问公网下载依赖包或者数据，则需要使用私有子网，同时配置一个 NAT 网关
- 如果您只需要访问 S3 和 EFS，不需要公网访问，则可以只配置 VPC 中的 S3 Endpoint，具体可以参考文档。

针对安全组参数security_group_ids，您需要确保绑定了此安全组的资源可以正常访问 EFS 文件系统，否则会出现文件系统挂载失败的情况，导致训练无法进行。
