function theta_out = mapping(theta_target, current_theta, directions)
    dtheta = directions.*(theta_target - current_theta);
    theta_out = rad2deg(dtheta) + [80 80 50 50 0 0];
end