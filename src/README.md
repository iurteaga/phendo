# Phendo src code

## Python dependencies

The src code requires the following python libraries:

    - https://numpy.org/
    - https://www.scipy.org/
    - https://matplotlib.org/
    - https://pandas.pydata.org/
    - https://amueller.github.io/word_cloud/
    - https://scikit-learn.org/stable/

## Data dependencies

The following src code assumes that ../data contains all the pre-processed data, ready to be analyzed. 

Specifically:

- ../data/q_ids_njdm contains the pre-processed data, in the following format:
    - f_emission.pickle: pickled dictionary indicating, for each question id, which distribution (multinomial) and its dimensionality
    - X.pickle: pickled numpy array indicating, for each participant (row) the data-source specific vocabulary id each data point (column) corresponds to
    - XD.pickle: pickled numpy array indicating, for each participant (row) the data-source id each data point (column) corresponds to
    - data_participant_ids: numpy array that maps each row (participant) in the preprocessed data to its corresponding phendo id
    - n_observations_matrix: numpy array that contains for each participant (row) the number of observations per data-source id (column)
    - n_days_matrix: numpy array that contains for each participant (row) the number of days per data-source id (column)

- ../data/q_ids_njdm/vocab describes the mapping from vocabulary id to phendo response
- ../data/q_ids_njdm/vocab_ordered describes a more intuitive ordering of phendo vocabulary responses

- ../data/werf_and_profile_data/profile.csv is the Phendo profile info file
- ../data/werf_and_profile_data/participant_dob.csv provides the dob of Phendo users
- ../data/werf_and_profile_data/werf_survey.csv contains WERF survey data
- ../data/werf_and_profile_data/phendoid_email_pid.csv' maps from PhendoID to email address used in WERF cuestionnaire

- ../data/q_ids_njdm/selected_participants/expert_groupings contains the phendo participant assignments of experts: each file must contain, per row (cluster), the participant ids assigned to this cluster (separated by commas)

## Phenotype data

Execute phenotyping, several realizations over 

- python3 phenotype_with_MixtureModel_multipleDataSources.py -data_dir ../data/q_ids_njdm -result_dir ../results

Execute phenotyping with train-test splits
- python3 phenotype_traintest_with_MixtureModel_multipleDataSources.py -data_dir ../data/q_ids_njdm -data_sources all -result_dir ../results -R 5 -traintest_ratio 0.8 -traintest_split balanced -split_n_obs 40

Once finalized, we can visualize inference results with

- python3 plot_results.py

## Evaluate phenotypes

To plot overall train-test phenotyping results 

- python3 eval_traintest.py

To figure out the associations between assignments and other information of interest

- python3 compute_associations.py

To select a random subset of participants and their data

- python3 select_participants.py

To assign participants to learned phenotypes

- python3 participant_assignments.py

To evaluate, given some expert grouping files, the confusion matrices with the learned phenotypes

- python3 eval_selected_participants.py

## SI clustering alternatives

- python3 clustering_alternatives.py
