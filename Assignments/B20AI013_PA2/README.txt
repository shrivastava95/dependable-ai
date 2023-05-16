Ishaan Shrivastava B20AI013
Dependable AI Assignment 2 - Bias and Explainability

-----------------------------------------------------------
This is a guide to the directory structure of this assignment including my work for this.



-----------------------------------------------------------
INSTRUCTIONS FOR RUNNING THE FILES

Question 1:
1. Load up a colab file and make sure all the required files are present in the appropriate places or edit the first part where the imports are being made or the files are being downloaded and moved around
2. run the cells one by one in a sequential order. If you encounter a memory limit error anywhere, just type torch.cuda.empty_cache() and this will probably perform some of the garbage collection.

Question 2:
1. Navigate to B20AI013_PA2_Q2/FACIL in terminal
2. Run `pip install -r requirements.txt`
3. Navigate to B20AI013_PA2_Q2/FACIL/scripts
4. Run `sh script_cifar100.sh lwf 0 base` to run the LwF part
5. Run `sh script_cifar100.sh bic 0 fixd` to run the BiC part
     -> make sure that, in B20AI013_PA2_Q2/FACIL/src/approach/bic.py, the line 278 and 279 are uncommented and 280 and 281 are commented before this
6. Run `sh script_cifar100.sh bic 0 fixd` to run the BiC-modified part
     -> make sure that, in B20AI013_PA2_Q2/FACIL/src/approach/bic.py, the line 278 and 279 are commented and 280 and 281 are uncommented before this



-----------------------------------------------------------
DIRECTORY STRUCTURE

The important files and folders are enumerated as follows:

-> README.txt     (YOU ARE HERE)                      > README file containing the instructions for running various parts of this assignment

-> B20AI013_PA2_demo.webm                             > Video demonstration containing the demo for how the various parts of the assignment work and are run

-> B20AI013_PA2_Q1                                    > Folder containing every part of question 1
       => B20AI013_PA2_Q1_code                        > Colab file containing all the code for question 1
       => CrisisMMD_v2.0                              > Folder containing the CrisisMMD dataset data
              +> annotations                          >
              +> crisismmd_datasplit_all.zip          >
              +> data_image                           >
              +> json                                 >
              +> .DS_Store                            >
              +> crisismmd_datasplit_all              >
              +> Readme.txt                           > Readme for the CrisisMMD dataset files/folders
-> B20AI013_PA2_Q2                  
       => FACIL                                       > Folder containing every part of question 2
              +> data                                 > Folder containing CIFAR-100 dataset
              +> docs                                 > 
              +> results                              > Folder containing the results for LwF, BiC, BiC-modified
                     #> cifar100_icarl_bic_fixd_0     > Folder containing the results for BiC and BiC-modified (both are together cause of a bug, sorry)
                     #> cifar100_icarl_lwf_base_0     > Folder containing the results for LwF
              +> scripts                              > Folder containing the scripts for running the experiments
                     #> script_cifar100               > Folder containing the script for CIFAR100 experiments. run this script as per the instructions.
              +> src                                  > 
                     #> approach                      > Folder containing the code implementations for different CIL approaches
                            *> lwf.py                 > .py containing the code for LwF
                            *> bic.py                 > .py containing the code for BiC and BiC-modified
              +> environment.yaml                     > 
              +> LICENSE                              > 
              +> README.md                            > 
              +> requirements.txt                     > 





