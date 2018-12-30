# GeneticAlgo-Neural-Network
In this project I have implemented the Genetic Algorithm in Python .
The project involves a series of steps including ,
Normalization of data,
binarization, 
Crossover,
Mutation and 
Debinarization. 


The data set has been randomly divided into test data and training data. 

At the end I have also plotted a 3d scatter plot along with finding Overall Error .


About the dataset ….

• The dataset is associated to body fat percentages in human body.

• Each column has a label on top.

• The columns show the circumference measurement (cm) of various parts of body.

• The last column shows the target ( y ) values, which are BodyFat in percentage.




Step 1 : Fitness Evaluation

• Fitness function shows how good is a chromosome as solution to your
problem. Its mainly user-defined function. 

• 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑉𝑎𝑙𝑢𝑒 of all the chromosomes generated are calculated and are assigned to them. This is done before you binarize them. 




Step 2 : Selection

• In this step we select ‘fittest’ parent from the existing population already created, in order to produce two offsprings with every
other member of population.

• The fittest means the chromosome which show highest value from Eq(3)




Step 3: Cross Over (mating)

• The most common type of Cross-Over is single point crossover.

• In single point crossover, you choose a point at which you swap the remaining bits(genes) from one parent to the other.

• The illustration in this slide help you understand it visually.

• As you can see, the offspring takes one section of the chromosome from each parent.

• The point at which the chromosome is broken depends on the randomly selected crossover point.

• This particular method is called single point crossover because only one crossover point exists. Sometimes only child 1 or child 2 is created, but oftentimes both offspring are created and put into the new population.




Step 4: Mutation

• In mutation you randomly select - let’s say – 5 % of the bits in the chromosome, and flip them to 0 if they are 1, and flip them to 1 if they are zero.

• These bits need not to be beside each other.


