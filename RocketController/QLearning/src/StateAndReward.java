public class StateAndReward {

	private static final int angleStates = 10;
	private static final int horStates = 4;
	private static final int vertStates = 6; 
	private static final double MAX_ANGLE = 1; 
	private static final double MIN_ANGLE = -1; 
	private static final double MIN_VX = -0.7;			 
	private static final double MAX_VX = 0.7;			 
	private static final double MIN_VY = -0.3;
	private static final double MAX_VY = 0.3;
	/* State discretization function for the angle controller */
	public static String getStateAngle(double angle, double vx, double vy) {

		/* TODO: IMPLEMENT THIS FUNCTION */
		
		//discretize over angle top get an state
		return "an"+discretize2(angle, angleStates, MIN_ANGLE,MAX_ANGLE)+";";
	}

	/* Reward function for the angle controller */
	public static double getRewardAngle(double angle, double vx, double vy) {

		/* TODO: IMPLEMENT THIS FUNCTION */
		
		//calculating the reward based on angle of rocket.
		return Math.abs(angle) <= MAX_ANGLE ? 1-Math.abs(angle)/MAX_ANGLE : 0;
	}

	/* State discretization function for the full hover controller */
	public static String getStateHover(double angle, double vx, double vy) {
		
		/* TODO: IMPLEMENT THIS FUNCTION */
		
		//discretize over angle, vx and vy to get the states
		var discreteAngle = discretize(angle, angleStates,MIN_ANGLE,MAX_ANGLE);
		var discreteVx = discretize(vx, horStates, MIN_VX, MAX_VX);
		var discreteVy = discretize(vy, vertStates, MIN_VY, MAX_VY);

		//return an state
		return "an"+discreteAngle+"vx"+discreteVx+"vy"+discreteVy+";";

	}

	/* Reward function for the full hover controller */
	public static double getRewardHover(double angle, double vx, double vy) {

		/* TODO: IMPLEMENT THIS FUNCTION */
		
		
		//calculating the reward for goal position, the closer it will be to goal position the more the reward.
		double rewardA = 0;
		double rewardH = 0; 
		double rewardV = 0;
		
		//calculating the reward for each function
		rewardA = Math.abs(angle) <= MAX_ANGLE ? 1 - Math.abs(angle)/MAX_ANGLE : 0 ;
		rewardH = Math.abs(vx) <= MAX_VX ? 1 - Math.abs(vx)/MAX_VX : 0 ;
		rewardV = 1.5*(Math.abs(vy) <= MAX_VY ? 1 - Math.abs(vy)/MAX_VY : 0) ;
		
		// squaring the reward to delete the chances of negative reward.
		return Math.pow(rewardA,2) + Math.pow(rewardH,2) + Math.pow(rewardV,2);
		
	}

	// ///////////////////////////////////////////////////////////
	// discretize() performs a uniform discretization of the
	// value parameter.
	// It returns an integer between 0 and nrValues-1.
	// The min and max parameters are used to specify the interval
	// for the discretization.
	// If the value is lower than min, 0 is returned
	// If the value is higher than min, nrValues-1 is returned
	// otherwise a value between 1 and nrValues-2 is returned.
	//
	// Use discretize2() if you want a discretization method that does
	// not handle values lower than min and higher than max.
	// ///////////////////////////////////////////////////////////
	public static int discretize(double value, int nrValues, double min,
			double max) {
		if (nrValues < 2) {
			return 0;
		}

		double diff = max - min;

		if (value < min) {
			return 0;
		}
		if (value > max) {
			return nrValues - 1;
		}

		double tempValue = value - min;
		double ratio = tempValue / diff;

		return (int) (ratio * (nrValues - 2)) + 1;
	}

	// ///////////////////////////////////////////////////////////
	// discretize2() performs a uniform discretization of the
	// value parameter.
	// It returns an integer between 0 and nrValues-1.
	// The min and max parameters are used to specify the interval
	// for the discretization.
	// If the value is lower than min, 0 is returned
	// If the value is higher than min, nrValues-1 is returned
	// otherwise a value between 0 and nrValues-1 is returned.
	// ///////////////////////////////////////////////////////////
	public static int discretize2(double value, int nrValues, double min,
			double max) {
		double diff = max - min;

		if (value < min) {
			return 0;
		}
		if (value > max) {
			return nrValues - 1;
		}

		double tempValue = value - min;
		double ratio = tempValue / diff;

		return (int) (ratio * nrValues);
	}

}
