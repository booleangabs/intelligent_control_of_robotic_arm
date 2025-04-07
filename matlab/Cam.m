classdef Cam
    properties
        cam
        cameraHeight
    end

    methods
        function obj = Cam(cameraHeight)
            obj.cam = webcam;
            obj.cameraHeight = cameraHeight;
        end


        function [real_pos, obj_positions, aruco_detected] = detect_aruco(obj, plot_enabled)
            cam = webcam;
            
            aruco_detected = false;
        
            frame = snapshot(cam);
            grayFrame = rgb2gray(frame);
            

            %grayFrame = imresize(grayFrame, [480, 640]);
            %img_smooth = imgaussfilt(grayFrame, 2);

        
            real_pos = [];
            obj_positions = [];
        
            try
        
                if plot_enabled
                    imgMarked = frame;
                    
                end
                [markerId, markerLocation, detectedFamily] = readArucoMarker(grayFrame);
        
                if length(markerId) < 2
                    error("No aruco  detected!");
                end
                
                
                markers = {[], []};
                robotTheta = 0; % Inicializa ângulo do robô
                for i = 1:length(markerId)
                    if markerId(i) ~= 1 && markerId(i) ~= 0
                        continue;
                    end
                    
                    corners = markerLocation(:,:,i);
                    d1 = norm(corners(1,:) - corners(2,:));
                    d2 = norm(corners(2,:) - corners(3,:));
                    d3 = norm(corners(3,:) - corners(4,:));
                    d4 = norm(corners(4,:) - corners(1,:));
                    
                    % Validação geométrica
                    if abs(d1 - d3) > 5 || abs(d2 - d4) > 5
                        continue;
                    end
        
                    position = mean(corners, 1);
                    markers{markerId(i) + 1} = [markers{markerId(i) + 1}; position];
                    
                    if plot_enabled
                        % Desenha marcadores
                        polygonVector = reshape(corners', 1, []);
                        imgMarked = insertShape(imgMarked, 'Polygon', polygonVector, 'Color', 'green', 'LineWidth', 2);
                        %imgMarked = insertText(imgMarked, position, sprintf('ID: %d', markerId(i)), 'FontSize', 18, 'BoxColor', 'green');
                    end
                end
                
                % Cálculo de distância e posição relativa
                if ~isempty(markers{1})
                    aruco_detected = true;
                    pos0 = markers{1};
                    pos0 = pos0(1,:);
                    pos = img2real(pos0, obj.cameraHeight);
                    real_pos  =pos;
                    positions_obj = markers{2};
                    for i = 1:size(markers{2},1)
                        obj_positions = [obj_positions; img2real(positions_obj(i,:), obj.cameraHeight)]; 
                    end
                    
                    
                    % Exibe informações
                    textPos = pos0;
                    imgMarked = insertText(imgMarked, textPos + [0, 30], sprintf('X: %.2f m, Y: %.2f m', pos(1), pos(2)), 'FontSize', 18, 'BoxColor', 'blue');
                end
                
            catch ME
                if plot_enabled
                    imgMarked = insertText(imgMarked, [10, 10], 'Erro ou nenhum marcador', 'FontSize', 18, 'BoxColor', 'red');
                end
            end
            if plot_enabled
                imshow(imgMarked);
                drawnow;
                
                clear cam;
            end
            
        end
        
    end
end
function pos = img2real(imgpos, cameraHeight)
    focalLength = 1500; 
    real_pos_x = -(cameraHeight * imgpos(1)) / focalLength;
    real_pos_y = (cameraHeight * imgpos(2)) / focalLength;
    pos = [real_pos_x real_pos_y];
end