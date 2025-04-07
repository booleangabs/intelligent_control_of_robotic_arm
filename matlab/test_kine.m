l1 = 10 / 100;
l2 = 12.6 / 100;
l3 = 6 / 100;
l4 = 3;
L(1)=Revolute('d', l1, 'a', 0, 'alpha', pi/2);
L(2)=Revolute('d', 0, 'a', l2, 'alpha', 0);
L(3)=Revolute('d', 0, 'a', 0, 'alpha', -pi/2);
L(4)=Revolute('d', l3, 'a', 0, 'alpha', pi/2);
L(5)=Revolute('d', 0, 'a', 0, 'alpha', pi/2);
L(6)=Revolute('d', l4, 'a', 0, 'alpha', 0);
AngleOffset=[0 pi/2 -pi/2 0 pi 0];
r=SerialLink(L,'name','6DOF Manipulator Arm','offset',AngleOffset);

T = r.fkine(deg2rad([100 120 20 10 0 0]));
%q = rad2deg(ikine_custom(T, l1, l2, l3));

vpa(T,2)
%vpa(q, 2)