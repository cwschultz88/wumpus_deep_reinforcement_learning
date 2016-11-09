# wumpus_deep_reinforcement_learning

This repo is set up to apply the principles of Deep Q Learning outlined in:
	
Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).

to the Wumpus Game. The Wumpus game rules can be found in the textbook "Artificial Intelligence: a Modern Approach" and also:

http://cis-linux1.temple.edu/~giorgio/cis587/readings/wumpus.shtml

Best Results:
The average score for the reinforcement agent was at 85 when the learning program crashed. This is after learning from about 25k games directly. It is important to note the score was still improving - the learning just stopped because the networked crashed and I did not implement parameter saving yet.... Just as a reference, my best hand crafted agents scored in this environment at:
  - Simple Reflex Agent: 160 - 180
  - Model Based Agent: 420 - 430
  - Search Based Agent: 380 - 400

So, the reinforcement approach still has a lot of learning left to compete against human developed AI. 

Future improvements to the code:
  - Stream line wumpus game code - taking too long to play a game
  - Apply convolutions to the deep Q network
  - Make the input into the deep Q network recurrent
  
The latter two improvements require more computation power than the local machine I have been using for this work.
