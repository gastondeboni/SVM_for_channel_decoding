function s = modulate_codewords(codewords,modulation_type)

    batch_size = size(codewords,1);
    m = strcmp(modulation_type,'BPSK')*1 + strcmp(modulation_type,'QPSK')*2 +strcmp(modulation_type,'8PSK')*3 + strcmp(modulation_type,'16QAM')*4;
    codewords_inreshape = codewords';
    codewords_dec = bit2int(codewords_inreshape(:),m);

    switch modulation_type
        case 'BPSK'
            s = 1-2*codewords_dec;
        case 'QPSK'
            s = pskmod(codewords_dec,4,'UnitAveragePower',true);
        case '8PSK'
            s = pskmod(codewords_dec,8);
        case '16QAM'
            s = qammod(codewords_dec,16,'UnitAveragePower',true);
    end
    s = reshape(s,[],batch_size).';
end