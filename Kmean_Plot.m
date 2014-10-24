figure(1)
for i=1:10000
    if results(i,3)==0
        hold on;
        plot(results(i,1), results(i, 2), '+b');
    elseif results(i,3)==1
        hold on;
        plot(results(i,1), results(i, 2), '+r');
    else
        hold on;
        plot(results(i,1), results(i, 2), '+g');
    end
    
end
figure(2)
for i=1:10000
    if results(i,4)==0
        hold on;
        plot(results(i,1), results(i, 2), '+b');
    elseif results(i,4)==1
        hold on;
        plot(results(i,1), results(i, 2), '+r');
    else
        hold on;
        plot(results(i,1), results(i, 2), '+g');
    end
end
