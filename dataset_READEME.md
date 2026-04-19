How the dataset is created: 

First, we decide the total number of players. In this case, we use 300 players. These players are divided into three equal groups. Each group represents a different type of gamer.
We use a random seed so that every time we run the code, we get the same results. This helps in testing and learning.
Each group is created using normal distribution. This means values are generated around an average number, with small changes so the data looks natural instead of perfectly fixed.

Types of players in the dataset: 

The first group is called Hardcore Grinders. These players spend a lot of time playing games every week. They have high skill in the game, shown by a high kill/death ratio. They do not spend much money and they are not very social in the game. They mostly play alone and focus on performance.

The second group is called Social Casuals. These players do not play many hours. Their skill level is average or low. They spend very little money in the game. But they are very active socially. They send messages, join teams, and interact with other players a lot.

The third group is called Whale Competitors. These players play a moderate number of hours. They have good skill in the game. The most important thing about them is that they spend a lot of money in the game. Their social activity is also moderate because they often play in teams or guilds.

Features in the dataset

Each player has four main features.Hours per week shows how much time a player spends playing the game. KD ratio shows how strong the player is in combat.
Monthly spending shows how much money the player spends in the game.
Social interactions shows how often the player interacts with others in the game.

How the data is generated : 

For each group, we use a normal distribution to generate values. Each feature has its own average and variation. After generating the values, we also apply limits so that no value becomes unrealistic like negative spending or extremely high values.

After creating all three groups, we combine them into one dataset. Then we shuffle the rows so the order is mixed and no group is visible directly.