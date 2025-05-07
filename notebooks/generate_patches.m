function NIFTI_NORDIC(fn_magn_in, fn_phase_in, fn_out, ARG)
    %  Scaling relative to the width of the MP spectrum, if one wants to be
    %  conservative
    %
    %  4/15/21 swapped the uint16 and in16 for the phase
    %
    %  VERSION 4/22/2021
    %  Copyright  Board of Regents, University of Minnesota, 2022

mag_data = randn(110, 110, 72, 13);
matdim = size(mag_data);
% kernel_size = [14 14 1];
kernel_size = repmat([round((size(mag_data, 4) * 11) ^ (1 / 3))], 1, 3);
w1 = kernel_size(1);
w2 = kernel_size(2);
w3 = kernel_size(3);
KSP_weight = mag_data(:,:,:,1) * 0;

patch_average_sub = 2;
patch_scale = 1;
patch_avg = 1;

KSP_processed = zeros(1, size(mag_data, 1) - kernel_size(1));

n1_arr = 1:size(KSP_processed, 2);
disp("n1")
disp(n1_arr)
for n1 = n1_arr
    KSP2_weight_tmp = KSP_weight([1:kernel_size(1)] + (n1 - 1), :, :, :);
    n2_arr = [1: max(1,floor(w2/patch_average_sub)):size(KSP2_weight_tmp,2)*1-w2+1  size(KSP2_weight_tmp,2)-w2+1];
    disp("n2")
    disp(n2_arr)
    for n2=n2_arr
        n3_arr = [1: max(1,floor(w3/patch_average_sub)):size(KSP2_weight_tmp,3)*1-w3+1  size(KSP2_weight_tmp,3)-w3+1  ];
        disp("n3")
        disp(n3_arr)
        for n3=n3_arr
            if patch_avg==1
                KSP2_weight_tmp(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) =...
                KSP2_weight_tmp(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) + patch_scale;
            else
                KSP2_weight_tmp(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) =...
                KSP2_weight_tmp(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +patch_scale;
            end
        end
    end

    KSP_weight([1:kernel_size(1)] + (n1 - 1), :, :, :) = KSP2_weight_tmp;
end
    return
