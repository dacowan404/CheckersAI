# CheckersAI
trains CNN using genetic algorithm and reinforcement learning


Checkers AI  
David Cowan  
CS 7375  
Spring 2022  

Abstract
In this project, I created a Checkers AI. This AI works by using a CNN to assign a score for each possible move. The move/board with the highest score is the move that the AI determines is the best. The weights for the CNN were trained by a genetic algorithm approach. Initially, the weights were set at random and competed against each other. The best weight set from each generation went on to create the next generation via recombination and mutation. After 15 generations, the weight sets seemed to converge and there was little genetic diversity. At this point training ended and the best weight set was found (Gen15_0111). This weight set was then test on the best from Gen0, Gen5, and Gen10 and was undefeated when it went first and had a winning record when it went second. I then played against Gen15_0111 and was able to beat it. While the AI generated from this project is not better than a human opponent, it was better than previous versions of itself and therefore with a few changes a much better AI can be generated. These changes include using a more complicated CNN architecture, creating a minimax search to determine the best move, and creating an end game database of optimal moves at the end of the game.

Introduction
Checkers is a board game that has been played for many centuries. Different cultures play differing versions of checkers. In America, the version most commonly is American checkers (also known as English draughts). In this version, two players compete on a board that is an 8x8 grid with 12 pieces on each side. The goal of the game is to capture all the opponent’s pieces or to leave the opponent with no legal move. Pieces can only move 1 diagonal move at a time unless it is possible to jump the opponent. Due to its’ rules, checkers is a deterministic, zero-sum, perfect information game.
Many different researchers have created checkers AI programs with varying degrees of success. The most successful checkers AI is known as Chinook which was created in 1989 by Dr. Jonathan Schaeffer and his team. Chinook has beaten the best humans in the world and was even used to prove a weak solution to checkers in 2007. This means that if both sides play perfectly the game will always result in a tie. While Chinook is the most successful checker AI ever, I decided to create an AI more similar to Blondie24.
Blondie24 is a checkers AI developed by David Fogel and Kumar Chellapilla in 1999 [1]. Blondie24 uses a minimax search with alpha-beta pruning and an evaluation function based on a convolutional neural network in order to pick the best move. The program used a genetic algorithm to determine the weights of the CNN and also used reinforcement learning to keep the best weight set and drop the worst the weight sets. During training, 15 parent weights sets were used to generate an offspring. Each of the 30 weights sets played 5 games with a scoring system (-2pt. for loss, 0 for tie, 1 for win). After playing the games, the 15 sets with the highest score were kept as parents to the next generation. After completing 840 generation, the best AI was tested on human opponents on The Zone website. The AI played a total of 165 games and achieved a rating of 2048 which put it in the top 1% for the website.
Inspired by Blondie24, the goal of this project is to develop a Checkers AI that is able to predict the best move by using a CNN. The weights of the CNN will be learning by using genetic algorithms and reinforcement learning. Each generation will start will 1028 weight sets and will select down to the best 16 which will be used to create the next generation. After several generations, I hope to create a checker AI that is very good at checkers.

Methodology
Environment and Tools
The AI was written and trained in Python3.9.7 in a Jupyter Notebook version 6.4.5. Additionally, Visual Studio Code was used to test against human opponents. The software uses Tensorflow and Numpy for making the CNN and updating the weights of the CNN. The Pandas package was used to keep track of weights and the generations. Random package was used when randomness was needed for the program (determining who move first, etc). Finally, imapaai-checkers package was used to help run checkers game (which moves are possible, making moves and updating board, etc).

CNN Architecture
The model used a CNN to score each possible move. Move with the highest score was selected as what the AI thought was the best move. The CNN consisted of 4 layers each with a sigmoid activation function (Table 1). The input to the CNN was a board of size 8x8x1. The values were 0 for empty tiles, 1 for regular pieces, and 2 for kings. The sign of the value denoted which player, positive numbers were used to denote self and negative numbers were used to denote opponent. The first layer of the CNN was a 2D Convolution Layer consisting of three 5x5 filters and had 0 padding. The output of this layer is a 4x4x3 matrix. The second layer was a 2D Convolution layer consisting of three 3x3 filters and had 0 padding. The output of this layer is a 2x2x3 matrix. This matrix was flattened into 12 nodes. The third layer is a fully connected layer with 6 nodes. Finally, the last layer is a fully connected layer with 1 node. This node represents the overall score of a move. Overall, the model consisted of 247 parameters (weights + bias).

![image](https://user-images.githubusercontent.com/43557995/198349668-7566bd49-abb4-4c6e-bb33-d1e093ebd45d.png)  
Table 1: CNN Architecture

Selection
Each generation started off with 1028 weight sets (referred to as individuals) which competed against each other to select the 16 best individuals. The first generation (gen0) was created by creating 2 arrays of size 247 populated with values randomly chosen between -1 and 1. These 2 individuals played against each other with the winner being added to gen0. Individuals that lost or tied were not kept. This process continued until 1028 individuals were added to gen0.
Initial selection consisted of random pairing up individuals in a head-to-head single elimination. When ties occurred both individuals were put in a tie pool. Individuals in the tie pool were randomly paired up and winner advanced. If a tie occurred for individuals in the tie pool, one of the two was randomly selected to advance. Overall, 3 rounds of single elimination were played to select the best 128 individuals in each generation.
These 128 individuals then competed in 4-member Round Robins. In these rounds, the individuals were randomly placed into groups of 4. Each member of the group played against every other member of the group and scored 3 points for a win and 1 point for a draw. No points were lost for a loss. After all the games were played the top 2 individuals from each group moved on to the next round. Overall, 3 rounds of Round Robins were completed in order to select the best 16 individuals in each generation. These 16 individuals were then used to create the next generation of individuals.
 In order to rank the 16 best individuals from a generation, a 16-member Round Robin was done. This was only done a few generations and did not affect how the next generation was produced. This Round Robin was done very similar to the 4-member Round Robin. Each individual played every other individual and 3 points were scored for a win, 1 point for a draw, and 0 points for a tie. Move order was determined at random. The number of points and wins were tracked for each individual. After completing all the games, the individuals were ranked based on the score with the number of wins resolving ties.


Generation
Generations started with the best 16 from the previous round except for Gen0 which was described in Selection section. These 16 individuals paired up with every other individual and used recombination to generate 2 new individuals. For recombination, a random integer, x, from 1-247 was generated. The two new individuals were made by concatenating individual A at location 0 to X with individual B at location X to 247 and by concatenating individual B at location 0 to X with individual A at location X to 247. For each generation, 240 individuals were made by recombination.
The 16 best individuals from the previous generation and the 240 new individuals were used to make even more individuals via mutations. Mutations occurred by adding an array made up of 247 values randomly selected from range -0.1 and 0.1 to a parent individual. Each of the 256 individuals at the end of recombination were used to generation 3 new individuals via mutation. For each generation, 768 new individuals were made which resulted in 1028 total individual in the new generation.
Each generation was saved as a CSV.

Training
In total, 15 generations were created. Since each generation was saved as a CSV, I was able to monitor the weights and see the progress of the training. After 3 generations, I played the best individual and won. After that, I continued training until I got to Gen12. I checked the CSV and saw that there were a couple of repeated sections among the best 16. However, it was not widespread and decided to continue training until Gen15. Upon viewing gen15, I notice my training had converged into a few models (Table 2). Due to the many highly similar individuals and very low genetic diversity I decided to end training.

![image](https://user-images.githubusercontent.com/43557995/198348922-4c3c4ce0-7e1c-4171-8f29-13695189b076.png)  
Table 2: Convergence of the individuals






Results
In order to find the best individual from Gen15, a 16-member Round Robin was done (Table 3). Gen15_0111 was found to be the best individual with 32 points and 9 wins (W:9, D:5, L:1).

![image](https://user-images.githubusercontent.com/43557995/198349959-411b2fbd-b8f9-4555-8d11-f64e25e7b125.png)  
Table 3: Ranking of models from Gen15

The best individual (Gen15_0111) was then tested against the best 16 from Gen0, Gen5, and Gen10 to test if the model had improved at playing checkers (Table 4). The best model played against each of the other individuals twice and alternated who went first. When the best model went first, it always won. When the other models went first, the best model still does well, and has a winning record (Gen0: 16-0-0, Gen5: 8-4-4, Gen10: 6-6-4). Overall, these results show that my methodology resulted in an individual that has learned how to better play checkers.

![image](https://user-images.githubusercontent.com/43557995/198350059-905d5638-e0fe-4e1f-9463-5999d4d25b68.png)  
Table 4: Gen15_0111 record against the 16 best from other generations.

I also played several games against the best individual (Gen15_0111) and won on every occasion with several pieces left. While the best individual may not be a challenge to most human opponents, this framework still resulted in better individuals over successive generations. With a few changes, I think a much better Checkers AI can be made. 

Discussion
Despite the AI that resulted from this project not being able to beat a human opponent, I feel that the idea worked and with a few changes a better AI can be made. One of the biggest changes to be made would be to make a more complex CNN model. The model created here is quite small and only consisted of 247 parameters. This is much smaller than the 5046 parameters trained for Blondie24 and most other CNNs which can have over a million parameters. So, I think increasing the model complexity and parameter count could lead to a much better AI model.
Another possible improvement is to incorporate a minimax search and use the CNN to evaluate each board a few moves out. Minimax search is commonly used in AI algorithms, especially in zero-sum games such as checkers. I was hoping to that my model would be able to learn the best move without having to use minimax to consider what the opponent would.  I now see that my simple model couldn’t learn that and would have benefited from using minimax search. With a much more complicated CNN model, skipping minimax may be possible, but this idea needs to be tested. However, a CNN as simple as mine would benefit from minimax search.
A third potential improvement could be to generate an endgame database to pick optimal moves for a win. This database would only be used when a win is only a few moves away. This type of database has been used to aid in other AI for a variety of games.
Lastly, I could change how generations are selected and generated. Before working on this project, I didn’t realize how common ties were in checkers, especially as skill increases. Therefore, some changes could be made on how single elimination rounds are done; for example, I could change how the tie pool is used. Additionally, I could change the format overall from a tournament style to a point-based style of selecting the best individuals from each generation. I also thought a little about changing how each generation is created but I am not sure what changes would benefit trainings.
Overall, I am proud of how my project went and that the AI generated was able to beat earlier versions of itself. I think with a few changes a much better checkers AI can be generated. The biggest changes I would make including making the CNN architecture more complex, developing a minimax search algorithm to determine moves, incorporating an end game database, and changing how selection of models is done.

Reference
1.	Chellapilla, Kumar; Fogel, David B. (1999). "Evolving Neural Networks to Play Checkers without Expert Knowledge" (PDF). IEEE Transactions on Neural Networks. 10 (6): 1382–1391. doi:10.1109/72.809083. PMID 18252639.
