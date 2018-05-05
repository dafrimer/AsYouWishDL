<h1>As You Wish Main Operations</h1>

Operations:

* Make sure door is closed*

* two concurrent actions
    * Motion Sensor - check if any recent activity.
        * if no: Keep going
        * If yes:  turn off the motion sensor begin camera
    * Audio sensor - check for the right mrow

* Begin Keras Model
    * start camera
        * save files alternating on each raspberry pi            
            * one will be local then scp'ed over to the other (multithreaded)
            * one will save local (multithreaded)
        * Remote worker Pi:
            * bashfile is looking for a txt file that starts the work
            * python uses pre-loaded keras model on every image then deletes
            * python concurrently send a message back to the pi if the image is the right cat

        * Head Pi
            * running against every file it saves on disk using pre-loaded keras model
            * if the right cat is detected continue

* Open Door
    * unlocks latch
    * opens door

* Lets cat in.
    * Begin distance sensor (making sure you dont ever close the cat inside)
    *  (*not set in stone)  Uses another camera inside that reassures its the right cat
    *  if it is. continue.  if not .. leave door open

    * Dispense food
        * use servo to open food lid.
        * begin weight sensor
            * while weight sensor is below the amount 2/3 of the amount keep dispensing.
        * servo closes food lid

        * servo passes food

    * if weight < certain_amount or distance is > certain amount.
        * retract food

    * if distance > certain amount close door

    
        

