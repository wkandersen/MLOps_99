# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16) (38%) coverage run --source=src -m pytest tests/ -> coverage report
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 99

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s224197, s224225, s220235

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

 We used the third-party framework Pytorch Image Models (timm) in our project. The framework provided us acces to pre-trained models including ResNet18 which we used as our base for our model. We loaded the pre-trained model using timm.create model and customized it to fit our dataset and classification problem. timm allowed us to efficiently implement transfer learning which significantly reduced the time we used to build and train the model. This meant that timm helped create a more accurate model, that was build faster than what we had previosly used. As timm made the process more effective it could have been an idea to use a more complex model to get a higher accuracy for example ResNet50.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

 We used a txt file for requirements to manage our dependencies. The list was auto-genereted by running "pip freeze > requirements.txt". Whenever new packages were installed, they were either added to the file manually or re-generated to ensure the file remained up to date. To get a complete copy of our development environment, you had to write "pip install -r requirements_dev.txt" in the terminal.
Additionally, most of the group also used Anaconda to keep track of all packages and dependecies, having a local environment dedicated to the project. For a brief period, we experimented with Dependabot to track dependecies however as a result of issues with compatibility and unintended changes to the codebase, we decided to remove it.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We used the cookiecutter template which was shown in the exercise modules because we thought it was quite fitting for the scope of our project. We have not modified it much from how it looks, but added some more folders for example the data_drifting and we also created a seperate folder under the src/group_99 for the api files. We didn’t make use of the data folder because the way our data was downloaded made more sense to just download it directly from kagglehub. The overall structure is the same, but we also added config.yaml files for use in wandb and hydra.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

 We implemented automated code quality and formatting rules using Pre-commit CI and GitHub Actions workflows. The pre-commit hooks ensure that our code met quality standards before being comitted while the GitHub Actions workflow were used for code formatting. We used the hooks to remove trailing whitespace, fix end-of-file issues, validate YAML files and to keep the requirements.txt file sorted and consistent. We used Ruff for the code formatting to check and automatically format the code on every push or pull request to the main brainch.

These concepts are important for larger project as they streamline the code making it easier to get an overview of the project. It also reduces errors as some errors are detected earlier than they would otherwise.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented six unit tests. One test validated the data file ensuring that it loads data correctly and that it retrieves images and labels as expected. The other five tests tested our model. Here it focused on the forward pass producing correctly shaped outputs, the training step computes valid losses, the validation and test executes without errors and the optimizer is configured properly.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of code is 25%, which includes all our source code. We are far from covering the complete code with this low percentage. Even with a 100% coverage we wouldn’t trust the code to be error free. This is because code coverage only covers whether the lines of code are executed during tests and not if the code behaves correctly in every scenario. Also high coverage does not guarantee that conducting a test is meaningful or that edge cases are handled correct. Errors could still happen from, for example unanticipated inputs. For instance, the code might pass all the tests but still fail in real-world scenarios as they can’t capture everything.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

 In our workflow we mostly used the main branch for making changes and then committing them directly. However, we occasionally created branches for specific tasks, such as optimizing existing code, and used pull requests to merge changes if the new implementation proved better. Using branches and pull request more systematically could probably have improved a lot on our version control process. If we for example had made a branch for each group member, we would probably have had fewer merge conflicts. The pull request would also have provided a better view of code changes and the collaborators, which would make debugging easier.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did use DVC, but had problems using it on Vertex AI, and therefore turned our dataset into a CSV file for the google cloud bucket aswell.
DVC tracks changes in data files. We could modify our datasets, or roll back to previous versions if needed. With dvc push and pull we could sync datasets without sharing large files. Linking dvc to our Google Cloud bucket, made it accessible for different machines and locations. So DVC makes it possible to track changes in data, in case we need to retrain the model on a updated dataset, and prevents inconsistencies like training a model on one dataset while deploying it with another.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration into four separate workflows: one for linting, one for pre-commit hooks, one for unit tests and one for Docker build and push.

The unit tests workflow are executed on three operating systems: ubuntu-latest, windows-latest and macos-latest. The workflow can be seen here: https://github.com/wkandersen/MLOps_99/actions/runs/12947205871

For the linting, we use Ruff, which is run on the ubuntu-latest environment.

For unit tests and linting the worflows include dependency caching enabled by pip cache as this reduces the test execution time. The workflows are triggered automatically on push and pull requests for the main branch ensuring that new code changes are validated immediately.

For the pre-commit hooks it was run on ubuntu-latest as the only operating system. The pre-commit framework was made to automate the tasks of trailing whitespace, fixing end-of-file errors, checking YAML syntax and updating requirements.txt. The hooks are triggered on each commit, helping to ensure consistent code formatting.

All three workflows are executed using Python 3.11.

The Docker workflow builds two images (application and API) using separate Dockerfiles, tags them with the Git commit SHA, and pushes them to Google Cloud Registry. It decodes a base64-encoded service account key from GitHub Secrets to authenticate with GCP. While dependency caching is used in other workflows, Docker layer caching is not yet enabled here. The workflow ensures images are validated and deployed only after builds.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

 We used Hydra to configure experiments, using a config file – config.yaml – to define our hyperparameters such as batch size, epochs and learning rate. The experiment is run by calling our train function with Hydra loading the necessary configurations automatically. It would work by writing the following: “python train.py hyperparameters.lr=0.01 hyperparameters.batch_size=32 hyperparmeters.epochs=10”.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

 To make sure that experiments are reproducible, we used Hydra which was also explained the last questions. In our code we loaded the hyperparameters with hydra by calling the `@hydra.main(config_path="config", config_name="config", version_base="1.3")` in the code. After this we could could call the hyperparameters located in the config.yaml file by using `hparams = config.hyperparameters`. Now all the hyperparameters defined in the config.yaml would be called by `hparams["name_of_hparam"]`, this ensured that any changes to the config file would be loaded into the code. This made it easy to reproduce the experiments as we could just change the hyperparameters in the config file and then run the code again.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

 ![wandb_logged_results](figures/<image>.<extension>)
As seen in the wandb logged results, we used wandb to track the results of our model. We tried using the hyperparameter sweep to check for different values of learning rate and batch size, but unfortunately we had issues with hydra and wandb integration. We therefore chose to just track the loss and accuarcy of the model to see how it performed. The loss and accuracy are important metrics to track as they inform us about how well the model is performing. The loss shows how well the model is learning and the accuracy shows how well the model is performing on the test data. By tracking these metrics we can see if the model is learning and if it is overfitting or underfitting. We also tracked the metric trainer/global_step which shows the numbers of steps the model has taken. It is important because it shows how many steps the model has taken to learn the data. It would have been interesting to also plot the validation accuaracy vs the training accuracy to see if the model was overfitting or underfitting. This would have been a good metric to track as it would show how well the model is generalizing to the test data and furthermore if we should try and adjust our dropout rate or other hyperparameters to improve the model.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

 Docker is very useful, for creating consistent and repoducible environments across development and deployment. We build two primary images:
Core application image, that handles data processing and model training. Built using a dockerfile, it installs dependencies, copies the codebase and configures the runtime environment.
API service image: Hosts a FastAPI server for predictions, exposes port 8000, and defines startup commands.
For cloud deployment, images are pushed to google cloud registry with tags tied to the git commit SHA, ensuring traceability.
The key benefits are: Reproducibility: identical environments in development, testing and production. Scalability: Easy deployment to cloud platforms like GCP. Isolation: speration of application and API dependencies.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

 We many times ran into issues that required debugging, and for fixing used a combination of different tactics which include print statements, the debugger in vs code and chatgpt. Alot of the issues also came from general setting up the model with things like input dimensions, layers and other similar problems. We have not made that much use of profiling in our code, which in hindsight would be something we would like to implement to especially highlight important parts of the code. This would be useful to see where the code is running slow and where we could optimize it or identify bottlenecks.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We successfully created a Google Cloud Storage Bucket and linked it to our project using Data Version Control (DVC), enabling efficient tracking and storage of our data. Additionally, we attempted to create a Virtual Machine (VM) instance to facilitate our workflows, but encountered issues when trying to enable the compute.googleapis.com API. Despite having the appropriate permissions and ownership role, we were denied access to enable the service. This appears to be a technical bug requiring further assistance from Google support to resolve.

Furthermore, we set up both an Artifact Registry and a Container Registry to streamline the management of Docker images and application dependencies for our project. These registries will allow us to build, store, and deploy our containerized applications efficiently in future stages of our work.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We attempted to utilize the Google Cloud Platform (GCP) Compute Engine service as the backbone for our project by creating a Virtual Machine (VM) instance. The goal was to use the VM for running our application, managing workloads, and facilitating tasks like training machine learning models and data preprocessing. However, we encountered an issue when trying to enable the compute.googleapis.com API, which is required to create and manage VM instances. Despite having appropriate ownership permissions, we were denied access to enable this service. This issue appears to be a bug, and we are seeking assistance from Google support to resolve it.

As a result, we were unable to proceed with creating or using any specific type of VM. However, our initial plan was to use an E2-standard VM instance, which offers a balance of cost and performance. This type of instance would have provided sufficient CPU, memory, and scalability to support our workloads efficiently. Once the issue with enabling the API is resolved, we intend to revisit this step and implement our VM usage as planned.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

 ![bucket_Q19](figures/<image>.<extension>)


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![artifact_Q20](figures/<image>.<extension>)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:



### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We explored using Compute Engine by attempting to create a virtual machine (VM) for training. Unfortunately, we were blocked from enabling the compute.googleapis.com API, even though we had owner permissions on the project. This prevented us from creating a VM instance for manual training.

We then turned to Vertex AI, turning our dataset into a csv file, as it didn’t seem to accept our DVC dataset. We didn’t want to let the Vertex AI make our model, but instead train our own model through Vertex, but we were once again denied access to enabling google apis, this time for artifact registry.

These issues with  API access prevented us from successfully training our model in the cloud. We are working to resolve these challenges and plan to retry the process once the necessary corrections are implemented.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

 We did manage to write an API for our model. We used FastAPI to do this and then used streamlit to create a frontend application that used our best model that we had trained and made a prediction of the uploaded image. Since our model validated to an accuracy of around 80% on the sea animals, we got quite good results from uploading pictures to the API. We did not really use the command line interface for uploading pictures because we had issues with it not working, so we found it was easier to make it a frontend application. This was then used for the prediction.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

 We deployed our API locally, but have also created a docker image for it in the cloud. As stated in the earlier question, we used a combination of FastAPI and streamlit to create the API application. The way to initiate it was by calling `uvicorn src.group_99.api.main:app` and then after that, you call `streamlit run frontend.py`, which would then open up to a website frontend where you can upload the file you want to predict. The API would then predict the image and give you the result of the prediction, by using the classes trained in the model. The classes were needed to be called by using the load_data function.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

 We did not perform any unit testing or load testing for our API. However, if we had we would have set up the unit testing using pytest to test individual components of FASTAPI endpoints. Here we would test the image preprocessing pipeline to ensure that the image files are correctly handled, and we would then test if they were correctly passed to the model.

We would have set up the load testing using locust to simulate many users interacting with our application. Here we would test how the API perform under varying loads such as sending requests from 100, 200, 1000 simultaneous user and then track metrics such as response time and failure rate.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring. We would like to have implemented monitoring for our deployed model such that we can track the model’s performance over time. Here we would measure different metrics such as error rates or prediction accuracy. By logging the metrics we would be able to identify any unexpected behavior, degradation in the model or shifts in data distribution.

We would also have monitored the memory usage to ensure the application is operating within the expected parameters. This would be integrated using Prometheus that would provide us with real-time insights and notifications, which would help maintain our deployed model and keep it reliable.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We spent 7kr in total on Google Cloud during the project. Most of it went towards repeated attempts to set up and configure Engine instances. Unfortunately we couldn’t get them fully working due to permission issues, even though we were listed as project owners. Second most was spent on cloud storage, which worked fine.
For future work in the cloud we would look forward to the clouds scalability and ease of access, (when granted access of cause). Problems like permission settings, became a major roadblock without direct support. We hit a wall trying to troubleshoot on our own, and all online help directed us to contacting google support, who haven’t answered yet.  that said, services like cloud storage seems efficient, quick, reliable and cost-effective. With help from google support we dream of exploring AI API’s pr serverless tools. For now it feel like an untapped potential for our proejct, and while having some learning-curve frustrations, we’re optimistic about its possibilities once we have overcome the initial barriers.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

 dd

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![mlops_overview.drawio](figures/<image>.<extension>)

The starting point in our main python application that downloads data to the local storage, as seen on the picture. The template is based on the one given during the exercises and intiated with the cookie cutter package. Our experiments are made reproducable by using the hydra config file system to make sure that the same hyperparamaters are used. We log our experiments by using wandb which logs to their website by using an API hidden in our github secrets enviroment. To make the python usable, we have conda enviroment which makes sure all the necessary requirements are there to make the code usable. After the model has been trained we have developed a frontend application that can be used by a user to upload a picture which is then predicted by the best model we have trained. We have used Git version control along data version control with google. Furthermore, we have implemented github actions to make sure that push/pulls are made correctcly and global variables are initiated correctly with example the wandb api. We have also implemented unittests that also make sure that the files work correctly. We've uploaded the dataset as well as our model code to our google cloud bucket. We've created workflows the create dockerfiles for the main applications as well as API.


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

 We had many struggles throughout the project, which started when we had a too big dataset for making small MVP as the subsetting we tried did not work as intended. Therefore we changed to a dataset that was more feasible to finish the project in time. Another big issue was the cloud computing which was not working as intended. Many small issues arised like hyperparameter sweep not working and api unit tests not working either, which we decided to not do to focus on the parts of project that were working and improve on them. We would've like to run the cloud computing to try and run a more in-depth test with more epochs and possibly run with different hyperparameters but unfortunately, we had issues with the API access which didn't allow us to use the cloud computing.

 We had problems with API enabling on Google Cloud, denying us access to use both Engine and Vertex. Besides when trying to set up the data for Vertex, the DVC files did not seem to be compatible with Google Cloud, forcing us to convert the data into a CSV file instead. When trying to link some of the cloud services to Git, the json keys seemed problematic, only after converting to base64 were we able to connect the two. Testing some of the Docker files turned out to be problematic too, as the storage amount available at some PCs wasn’t enough to handle the images.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

 Student s224225 was in charge of setting up the initial git repository and the initial cookie cutter project. Furthermore, the student was in charge of creating the model and train files. The student also set up the frontend api application.

 Student s220235 worked on workflows, specifically for dockerfiles, as well as DVC and Google Cloud. Setting up all the services, preparing work on the virtual machines (and getting permissions).





ChatGPT and GitHub Copilot was used in the code for debugging and inspiration to help develop parts of the code.
