from clearml import Task, TaskTypes
from clearml.automation import PipelineController

def create_pipeline():
    pipe = PipelineController(
        name='QA Training Pipeline',
        project='EduRegulation-Retrieval',
        version='1.0.0',
        add_pipeline_tags=True
    )

    pipe.add_step(
        name='preprocessing',
        base_task_project='EduRegulation-Retrieval',
        base_task_name='Preprocessing',
        parameter_override={
            'General/data_path': './data/finetune',
            'General/model_name': 'xlm-roberta-base'
        },
        execution_queue='default'
    )

    pipe.add_step(
        name='training',
        base_task_project='EduRegulation-Retrieval',
        base_task_name='Training',
        parameter_override={
            'General/model_name': 'xlm-roberta-base',
            'General/num_epochs': 3,
            'General/batch_size': 16,
            'General/learning_rate': 2e-5,
            'General/weight_decay': 0.01
        },
        parents=['preprocessing'],
        execution_queue='default'
    )

    pipe.add_step(
        name='testing',
        base_task_project='EduRegulation-Retrieval',
        base_task_name='Testing',
        parameter_override={
            'General/data_path': './data',
            'General/model_path': 'vodailuong2510/MLops'
        },
        parents=['training'],
        execution_queue='default'
    )

    pipe.add_step(
        name='evaluation',
        base_task_project='EduRegulation-Retrieval',
        base_task_name='Evaluation',
        parameter_override={
            'General/data_path': './data',
            'General/model_path': 'vodailuong2510/MLops'
        },
        parents=['testing'],
        execution_queue='default'
    )

    return pipe

def setup_tasks():
    preprocessing_task = Task.create(
        project_name='EduRegulation-Retrieval',
        task_name='Preprocessing',
        task_type=TaskTypes.data_processing
    )
    preprocessing_task.set_parameter('General/data_path', './data/finetune')
    preprocessing_task.set_parameter('General/model_name', 'xlm-roberta-base')
    
    training_task = Task.create(
        project_name='EduRegulation-Retrieval',
        task_name='Training',
        task_type=TaskTypes.training
    )
    training_task.set_parameter('General/model_name', 'xlm-roberta-base')
    training_task.set_parameter('General/num_epochs', 3)
    training_task.set_parameter('General/batch_size', 16)
    training_task.set_parameter('General/learning_rate', 2e-5)
    training_task.set_parameter('General/weight_decay', 0.01)

    testing_task = Task.create(
        project_name='EduRegulation-Retrieval',
        task_name='Testing',
        task_type=TaskTypes.testing
    )
    testing_task.set_parameter('General/data_path', './data')
    testing_task.set_parameter('General/model_path', 'vodailuong2510/MLops')

    evaluation_task = Task.create(
        project_name='EduRegulation-Retrieval',
        task_name='Evaluation',
        task_type=TaskTypes.testing
    )
    evaluation_task.set_parameter('General/data_path', './data')
    evaluation_task.set_parameter('General/model_path', 'vodailuong2510/MLops')

if __name__ == "__main__":
    setup_tasks()
    
    pipeline = create_pipeline()
    pipeline.start()