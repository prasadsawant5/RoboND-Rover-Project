import numpy as np
from navigation import rad_to_deg, deg_to_rad, to_polar_coords

def mean_nav_angle(Rover):
    if Rover.nav_angles.any() and len(Rover.nav_angles) > 0:
        angle = np.average(rad_to_deg(Rover.nav_angles), weights=Rover.nav_weights)
    else:
        angle = -15
    return angle


def set_angle(Rover):
    mean_angle = mean_nav_angle(Rover)
    if not np.isnan(Rover.wall_point).all():
        wall_normal_vect = rad_to_deg(Rover.wall_angle)
        weights = np.array([0.33, 0.66])
        angles = np.array([wall_normal_vect, mean_angle])
        angle = np.average(angles, weights=weights)
    else:
        angle = mean_angle

    return angle


def is_moving(Rover):
    if Rover.pos is None or Rover.last_pos is None:
        return True

    displacement = ((Rover.pos[0] - Rover.last_pos[0]) ** 2 +
                    (Rover.pos[1] - Rover.last_pos[1]) ** 2) ** 1/2
    if displacement > Rover.etoll_disp:
        return True
    else:
        return False


def see_sample(Rover):
    # Are we seeing a sample on our left side?
    if Rover.seeing_sample:
        return True
    return False


def check_data(Rover):
    if Rover.throttle == np.nan:
        Rover.throttle = 0
    if Rover.steer == np.nan:
        Rover.steer = 0
    if Rover.brake == np.nan:
        Rover.brake = 0


def track_sample(Rover):
    r_dists, r_rads = Rover.curr_sample_pos
    if Rover.near_sample:
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
    else:
        if Rover.vel < Rover.max_pursuit_vel and is_moving(Rover):
            # Set throttle value to throttle setting
            Rover.throttle = Rover.rock_pursuit_set
        else:  # Else coast
            Rover.throttle = 0

        if np.mean(r_dists) < 0.01:
            # slow down
            Rover.brake = Rover.slow_down_set
        else:
            Rover.brake = 0
        angle = rad_to_deg(r_rads)
        Rover.steer = np.clip(np.mean(angle), -15, 15)


def forward_mode(Rover):
    # Check the extent of navigable terrain
    if len(Rover.nav_angles) >= Rover.stop_forward:
        # If mode is forward, navigable terrain looks good
        # and velocity is below max, then throttle
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
            Rover.brake = 0
        elif Rover.vel > Rover.max_vel:
            Rover.throttle = 0
            Rover.brake = Rover.slow_down_set / 10
        else:  # Else coast
            Rover.throttle = 0
            Rover.brake = 0
        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(set_angle(Rover), -15, 15)

    # If there's a lack of navigable terrain pixels then go to 'stop' mode
    elif len(Rover.nav_angles) < Rover.stop_forward:
        # Set mode to "stop" and hit the brakes!
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = 'stop'


def stop_mode(Rover):
    # If we're in stop mode but still moving keep braking
    if Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
    # If we're not moving (vel < 0.2) then do something else
    elif Rover.vel <= 0.2:
        # Now we're stopped and we have vision data to see if there's a path forward
        if len(Rover.nav_angles) < Rover.go_forward:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
            Rover.steer = -15
        # If we're stopped but see sufficient navigable terrain in front then go!
        elif len(Rover.nav_angles) >= Rover.go_forward:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to mean angle
            Rover.steer = np.clip(set_angle(Rover), -15, 15)
            Rover.mode = 'forward'

def stuck_mode(Rover):
    if Rover.vel > Rover.reverse_vel_set:
        Rover.throttle = Rover.reverse_set
        Rover.brake = 0
        Rover.steer = 0
    elif Rover.throttle < 0 and not is_moving(Rover):
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0
        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
        Rover.steer = -15
        Rover.mode = "stop"
    else:
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = "stop"


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

