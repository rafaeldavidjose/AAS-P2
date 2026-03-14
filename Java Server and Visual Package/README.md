# 1. Installing Java
Make sure you have the correct java installation on your computer.
## Install **Java Runtime Environment 17 (JRE 17)**

The Pac-Man engine was updated to compile against **Java 17**, so you must run the JARs using a Java 17 runtime.
### **Recommended Downloads**
#### **Adoptium Temurin JRE 17 (Official & Free)**
https://adoptium.net/temurin/releases/?version=17&package=jre

Choose:
- OS: Windows / macOS / Linux
- Type: **JRE**
- Version: **17 LTS**
### **Azul Zulu JRE 17 (Alternative)**
https://www.azul.com/downloads/?package=jre&version=17
### Verify Installation
Run (in terminal):
`java -version

Should show:
`openjdk version "17.x.x"`

If you see:
`version "1.8" (or 52.0)`

You are running a *newer* version of Java 8 and the JAR files **will not work**. So make sure you are running an older version. 

# 2. Running Headless Game
The EnvServer.jar is the headless version of the game, meaning you can run 
quick simulations or rollouts using this version, without the visual component.
## Running the Server
Run the server in the windows terminal, making sure you are in the folder where the .jar is located.

Use the following command:
`java -jar EnvServer.jar` 

By default the server will run in port `5000`, if you want to change the port 
to run rollout simulations do the following:

`java -jar EnvServer.jar 5001`
This opens a new EnvServer in port `5001` allowing you to run simulations parallel to the actual game. 

Once the server is running you can then run your agent using the Python package through the `ms_pacman_env.py` script (see the examples).

# 3. Running the Visual Pacman Environment
The `PythonPolicyPacman.jar` provides a visual representation of the Ms. Pac-Man game allowing you to visualize gameplay with any agent algorithm. 

This works in conjunction with the `policy_server.py` script in the Python package. 

## Running the Server
First run the python script `policy_server.py` using the agent you want to test. If using an agent that requires simulations, start the `EnvServer.jar` first using port `5001`. 

Once the `policy_server.py` is running you run `PythonPolicyPacman.jar` using the following command:
`java -jar .\PythonPolicyPacMan.jar`

The visual component will start and the game will run with the associated agent. 
