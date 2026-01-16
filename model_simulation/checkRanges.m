function isInAnyRange = checkRanges(startIndices, endIndices, i)
    isInAnyRange = false;
    for j = 1:length(startIndices)
        if i >= startIndices(j) && i <= endIndices(j)
            isInAnyRange = true;
            break;
        end
    end
end