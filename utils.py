def avoid_obstacle(rob):
    rob.stopMotors()
    rob.moveWheelsByTime(-10,-10,2, wait=True)
    rob.moveWheelsByTime(-10,10,2, wait=True)