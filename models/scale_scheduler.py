def temperature_scheduler(step,max_step,tau,strategy = 'linear'):
    
    if strategy == 'linear':
        return tau - (tau-1)/(max_step-1)*(step-1)
    elif strategy == 'exp':
        return tau**((1-(step-1)/(max_step-1)))
    else:
        return tau