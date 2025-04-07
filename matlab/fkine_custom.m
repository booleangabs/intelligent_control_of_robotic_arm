function pos = fkine_custom(lengths, theta, z)
    % Extrai os ângulos
    theta1 = theta(1);
    theta2 = theta(2);
    theta3 = theta(3);
    theta4 = theta(4);
    
    % Cálculo da posição do efetuador final
    x = cos(theta1) * (lengths(1) * cos(theta2) + lengths(2) * cos(theta2 + theta3) + lengths(3) * cos(theta2 + theta3 + theta4));
    y = sin(theta1) * (lengths(1) * cos(theta2) + lengths(2) * cos(theta2 +  theta3) + lengths(3) * cos(theta2 + theta3 + theta4));

    % Retorna a posição cartesiana [x, y, z]
    pos = [x, y, z];
end