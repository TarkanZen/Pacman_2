**This is the Multiagent Search Algorithm Section of the Project.**

**All code structure rights belong to UC BERKLEY.**

**Code for the project can be found under multiAgents.py. Score: 25/25**

**Reflex Agent:**

python pacman.py -p ReflexAgent -l testClassic

python pacman.py --frameTime 0 -p ReflexAgent -k 1

python pacman.py --frameTime 0 -p ReflexAgent -k 2

python autograder.py -q q1

*Command above takes too long, the same test can be run without the graphics with the line below.*

python autograder.py -q q1 --no-graphics

**Minimax:**

python autograder.py -q q2

python autograder.py -q q2 --no-graphics

python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3

**Alpha-Beta Pruning:**

python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

python autograder.py -q q3

python autograder.py -q q3 --no-graphics

**Expectimax:**

python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10

python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10

python autograder.py -q q4

**Evaluation Function:**

python autograder.py -q q5

python autograder.py -q q5 --no-graphics
