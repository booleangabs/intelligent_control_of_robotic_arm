function pos = fkine(obj, theta, z)
    % Extrai os ângulos
    theta1 = theta(1);
    theta2 = theta(2);
    theta3 = theta(3);
    theta4 = theta(4);
    
    % Cálculo da posição do efetuador final
    x = cos(theta1) * (obj.l1 * cos(theta2) + obj.l2 * cos(theta2 + theta3) + obj.l3 * cos(theta2 + theta3 + theta4));
    y = sin(theta1) * (obj.l1 * cos(theta2) + obj.l2 * cos(theta2 +  theta3) + obj.l3 * cos(theta2 + theta3 + theta4));

    % Retorna a posição cartesiana [x, y, z]
    pos = [x, y, z];
end