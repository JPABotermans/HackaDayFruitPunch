# Hackaday Rules
_agent: combination of preprocessor and trained algorithm_
This document will specify the rules of the Hackaday, explain how your agent will be tested and how you have to hand in your agent. If there is at any time doubt about the rules, judging or submission consult the technical team. FruitPunch AI has the final say about what is and what is not a according to the rules.

## Rules

- Your agent must be written in python.
- While outside resources may be consulted, your agent must be created and trained by your team and thus cannot be blatantly copied from an outside source.
- You can only use the environment provided by FruitPunch AI or the hardware you brought to the Hackaday to train your algorithm.
- Your may not make use of the _info_ field provided by openAI Gym to pick an action during both evaluation and training.
- Your algorithm must make use of machine learning and can thus not be a so called 'Rational Agent' that has pre-programmed rules to decide on actions.
- Your algorithm must pick the action that is output by your agent
  - As an extension your preprocessor cannot pick the action that is output by your agent. This does not mean your preprocessor cannot perform complex tasks like for example generate a collection of heat maps from the pixel data.
- Your agent must be compatible with the _pycman_ framework.
  - This does not mean you have to use _pycman_ for creating and training your algorithm but we will use _pycman_ to load and evaluate your agent so it must adhere to the interface. More on this interface below.
- your agent must be able to play 50 games in under 10 minutes on the environment provided by FruitPunch AI.

## Judging
To evaluate your agent we will run it in _pycman_ with the following ini options:
- \[GAME]
  - compressor = DataIO
  - fps = 0
  - game_name = SpaceInvaders-v0
  - render = False
  - save_folder = data
  - session_name = \[your session name]
  - store_data = False
  - store_video = False
  - checkpoint_interval = 0
- \[Algorithm]
  - algorithm_name = \[your algorithm name]
  - preprocessor = \[your prepreocessor name]
  - session_restore = True
  - can_train = False
  - nr_evaluation = 50
  - condition_nr_steps = inf
  - condition_nr_games = 0
  - condition_ram_used = 95
- \[AlgorithmSettings]
  - options for your algorithm
- \[PreprocessorSettings]
  - options for your preprocessor

We rank the agents based on the average score after 50 games. If the two agents with the highest scores are within 10 points of each other the agents are tested another 50 games. The winner will then be determined by the highest score of all 4 runs.

## Deliverable

You must provide the tech team with the following files before the announced deadline:

- Your algorithm in the form of a python file that works with _pycman_.
- if you have one, your preprocessor in the form of a python file that works with _pycman_.
- all files your algorithm needs to load its trained state using the *_load_checkpoint* method.
- an ini file with the options to run your algorithm (especially the \[AlgorithmSettings] and \[PreprocessorSettings] sections are important).
- a small writeup explaining what strategy your agent uses, this does not need to be very elaborate.

## Algorithm interface

In order to work with _pycman_ your algorithm should implement the following methods:

- ```__init__(self, settings, *args, **kwargs)```: This method is called when initializing your algorithm, you can for example set your key word arguments here. This methods should call ```super().__init__(**settings)```. (if you are not using pycman during development we recommend commenting the super line out.)
  - ```settings```: the _GAME_ and _Algorithm_ settings as parsed from the ini file
  - ```**kwargs```: the keyword arguments as parsed from the _AlgorithmSettings_ section in the ini file
- ```_pick_action(self, observation)```: This method picks the action (an integer in the range \[0-5]) your agent will take during the training phase
  - ```observation```: the output of your preprocessor
- ```_pick_eval_action(self, observation)```: his method picks the action (an integer in the range \[0-5]) your agent will take during the evaluation phase
  - ```observation```: the output of your preprocessor
- ```_train```: trains your algorithm after collecting data (playing games) during the training phase
  - ```observations```: List of preprocessed observations (the same observations that were passed to the ```_pick_action``` method)
  - ```metadata```: a pandas dataframe containing the following info/columns:
    - ```step_id```: id of the step in the corresponding game
    - ```game_id```: id of the game
    - ```running_session_id```: id of the current session
    - ```reward```: reward recieved this step
    - ```done```: if the game is done on this step (indicates the last step of the current game)
    - ```info```: a dictionary with info about the current game that gym provides. In Space Invaders this dictionary only includes the single key value pair: {```ale.lives```: _an int containing the number of lives left_}
    - ```action```: what action was picked by your algorithm
  - ```_load_checkpoint(self, path)```: method that saves all files that are needed to load the current training state of your algorithm (for example network weights if using a neural net) *to the provided path*
    - ```path```: the path the method should write the files to
- ```_store_checkpoint(self, path)```: method that loads a trained state of your algorithm from files in the given directory
  - ```path```: the path the method should load the files from

## Preprocessor interface

in order to work with _pycman_ your preprocessor should implement the following methods:

- ```__init__(self, settings, *args, **kwargs)```: This method is called when initializing your preprocessor, you can for example set your key word arguments here. This methods should call ```super().__init__(**settings)```. (if you are not using pycman during development we recommend commenting the super line out.)
  - ```settings```: the _GAME_ and _Algorithm_ settings as parsed from the ini file
  - ```**kwargs```: the keyword arguments as parsed from the _PreprocessorSettings_ section in the ini file
- ```preprocess(self, observation)```: convert the pixeldata from the Space Invaders frame to data to input in your algorithm
  - ```observation```: a numpy array of shape (210, 160, 3) containing values in the range \[0-255] that represent the pixels of a 210x160 frame from Space Invaders. Note that every pixel has 3 values namely one for each color channel (RGB)
- ```get_output_size(self)```: returns the shape of your preprocessor output

## Extra info

- Do *NOT* quit Jupyter Lab, this will halt the docker container and we have no way to restart it. If you do so your team will be without an environment for the duration of the day.
- Each environment on IBM's server has access to a lot of threads but single thread performance is normal so a lot of speed can be gained by employing multi-threading.
- The environment has access to a Nvidia Tesla V100 but two environments share a single GPU and thus your container is capped to 50% of the cards capacity which should still be sufficient.

## Good luck and have fun with the hackaday!