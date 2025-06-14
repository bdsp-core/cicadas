function y = fcnSmoothPulse(t, pulseAmp, pulseMu, pulseWidth, pulseC);

    % Ensure x is non-negative for well-defined behavior
    t = max(0, t);
    % Calculate b parameter to ensure peak at mu
    b = pulseC / (pulseMu * pulseWidth);
    % Calculate the function value
    y = t.^pulseC .* exp(-b * t);
    % Normalize to have peak of amplitude a
    y = y / max(y) * pulseAmp;
end