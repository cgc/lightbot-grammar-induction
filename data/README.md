# Data format

We make the experiment data available as the CSV file [programs.csv](programs.csv) in this directory. The original experiment output, which provides more detailed information about program editing, is available [here](../experiment/data/human_raw/0.4/trialdata.csv).

## CSV format

Each row corresponds to a task a participant encountered. As noted in the paper, participants each solved 10 tasks in the same order.

| Column | Description |
|---|---|
| participant_idx | A unique identifier for a participant. |
| mdp_idx | A unique identifier for a task, corresponding to order encountered in the experiment. |
| skipped | A boolean column that indicates whether the participant skipped the task. |
| elapsed_minutes | The number of minutes a participant took to solve the task. |
| program_evaluations | The number of program evaluations a participant took. |
| program | The program a participant submitted. |
| preprocessed_program | A preprocessed version of the program a participant submitted, used for most analyses in the text. |
| programming_exp | Participant response to "How much experience do you have with computer programming?" |
| programming_game_exp | Participant response to "Have you played Lightbot or another similar programming game before?" |

## Program format

Programs consist of subroutines: the main subroutine where execution begins, and four others numbered from 1--4. Subroutines are separated by the pipe character ("|") and each consist of a sequence of single-character instructions. A table with the meaning of the instructions are below.

| Instruction | Description |
| --- | --- |
| S | Activate Light |
| W | Walk |
| J | Jump |
| L | Turn Left |
| R | Turn Right |
| 1 | Execute Subroutine 1 |
| 2 | Execute Subroutine 2 |
| 3 | Execute Subroutine 3 |
| 4 | Execute Subroutine 4 |

Here is an example program corresponding to one of the recurring examples presented in the paper (in Figures 1, 2, and 4). This program has a single subroutine (Walk, Walk, Walk, Light) that is called three times.

```
1LJL1RJR1|WWWS|||
```
