from clearml import PipelineController

pipe = PipelineController(
    project='EduRegulation-Retrieval',
    name='EduRegulation-Retrieval Pipeline',
    version='1.0',
)

pipe.add_step(
    name='data_preparation',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Data Preparation',
)

pipe.add_step(
    name='preprocessing',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Preprocessing',
    parents=['data_preparation'],
    parameter_override={
        'input_data_url': '${data_preparation.artifacts.processed_data.url}',
    },
)

pipe.add_step(
    name='vector_db_build',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Vector Database Build',
    parents=['preprocessing'],
    parameter_override={
        'processed_data_url': '${preprocessing.artifacts.cleaned_data.url}',
    },
)

pipe.add_step(
    name='model_definition',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Model Definition',
    parents=['vector_db_build'],
    parameter_override={
        'vector_db_url': '${vector_db_build.artifacts.vector_db.url}',
    },
)

pipe.add_step(
    name='training',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Training Model',
    parents=['model_definition'],
    parameter_override={
        'vector_db_url': '${vector_db_build.artifacts.vector_db.url}',
        'model_url': '${model_definition.artifacts.model.url}',
    }
)
pipe.add_step(
    name='testing',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Testing',
    parents=['training'],
    parameter_override={
        'trained_model_url': '${training.artifacts.trained_model.url}',
    }
)
pipe.add_step(
    name='evaluation',
    base_task_project='EduRegulation-Retrieval',
    base_task_name='Evaluation',
    parents=['testing'],
    parameter_override={
        'test_result_url': '${testing.artifacts.test_result.url}',
    }
)

pipe.start(queue='default')
pipe.wait()
pipe.stop()
