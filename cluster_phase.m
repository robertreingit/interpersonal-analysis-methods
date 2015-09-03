function rho_group =  cluster_phase(theta_i)

    [N,n] = size(theta_i);
    
    q_s_ti = sum(exp(1i*theta_i),2)/n;
    q_ti = angle(q_s_ti);
    
    theta_k_ti = theta_i - repmat(q_ti,1,n);
    theta_s_k = 1/N*sum(exp(1i*theta_k_ti));
    theta_k_hat = angle(theta_s_k);
    
    rho_group_i = abs(1/n*sum(exp(1i*theta_k_ti-repmat(theta_k_hat,N,1)),2));
    rho_group = mean(rho_group_i);    

end