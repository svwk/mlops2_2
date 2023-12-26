from clearml import Task
from clearml.automation import (HyperParameterOptimizer, UniformIntegerParameterRange)


task = Task.init(project_name='mlops2_2', task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)
args = {'template_task_id': "950e320154054bb189bd31220e72f9d8", 'run_as_service': False, }

an_optimizer = HyperParameterOptimizer(base_task_id=args['template_task_id'],
    hyper_parameters=[UniformIntegerParameterRange('General/n_estimators', min_value=60, max_value=120, step_size=20),
        UniformIntegerParameterRange('General/max_depth', min_value=2, max_value=7, step_size=1)],
    objective_metric_title='smape', objective_metric_series='smape', objective_metric_sign='min', )

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()
