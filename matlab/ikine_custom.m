function theta_target = ikine_custom(x_target, l1, l2, l3)
    %sphi = x_target(3)/(x_target(3)^2 + x_target(1));
    %cphi = x_target(1)/(x_target(3)^2 + x_target(1));

    phi = atan2(x_target(3), abs(x_target(1)));

    phi = deg2rad(10);

    cphi = cos(phi);
    sphi = sin(phi);

    theta1 = atan2(x_target(2), x_target(1));

    new_x = sqrt(x_target(1)^2 + x_target(2)^2);

    x2 = new_x - l3 * abs(cphi);
    z2 = x_target(3) - l3 * sphi;

    c3 = (x2^2 + z2^2 - l1^2 - l2^2)/(2*l1*l2);

    theta3 = -acos(c3);
    s3 = sin(theta3);

    % Calcular theta2
    c2 = ((l1 + l2*c3)*x2 + l2*s3*z2)/(x2^2 + z2^2);
    s2 = ((l1+l2*c3)*z2 - l2*s3*x2)/(x2^2 + z2^2);

    theta2 = atan2(s2, c2);

    theta4 = phi - (theta2 + theta3);

    theta_target = [theta1 theta2 theta3 theta4 0 0];
end