SP = {}
OKD = {}
ADW = {}

PROBLEMS = {"SP": SP, "OKD": OKD, "ADW": ADW}

SP["uniform"] = {                         # sample      init        reg?
    "t-0/0/0":  "Exp-20220116-091502-SP", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  "Exp-20220116-091645-SP", # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220117-090848-SP", # pi^t	    pi^0        0
    "t/0/r":    "Exp-20220117-090813-SP", # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220116-112702-SP", # pi^0	    pi^0        0
    "0/0/r":    "Exp-20220116-112733-SP", # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220116-112813-SP", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  "Exp-20220116-112836-SP", # pi^t	    pi^0-pi^t   reg
}

SP["2018011309"] = {                      # sample      init        reg?
    "t-0/0/0":  "Exp-20220114-121504-SP", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  "Exp-20220114-121550-SP", # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220114-165316-SP", # pi^t	    pi^0        0
    "t/0/r":    "Exp-20220114-121419-SP", # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220114-165201-SP", # pi^0	    pi^0        0
    "0/0/r":    "Exp-20220114-165356-SP", # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220114-120904-SP", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  "Exp-20220114-121237-SP", # pi^t	    pi^0-pi^t   reg
}

SP["19283746"] = {                        # sample      init        reg?
    "t-0/0/0":  "Exp-20220114-144437-SP", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  "Exp-20220114-144507-SP", # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220114-144633-SP", # pi^t	    pi^0        0
    "t/0/r":    "Exp-20220114-144727-SP", # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220114-165757-SP", # pi^0	    pi^0        0
    "0/0/r":    "Exp-20220114-165902-SP", # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220114-145015-SP", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  "Exp-20220114-165454-SP", # pi^t	    pi^0-pi^t   reg
}

SP["20000308"] = {                        # sample      init        reg?
    "t-0/0/0":  "Exp-20220114-165537-SP", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  "Exp-20220114-165614-SP", # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220114-185932-SP", # pi^t	    pi^0        0
    "t/0/r":    "Exp-20220114-165945-SP", # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220114-210922-SP", # pi^0	    pi^0        0
    "0/0/r":    "Exp-20220114-185747-SP", # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220114-185824-SP", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  "Exp-20220114-185852-SP", # pi^t	    pi^0-pi^t   reg
}


OKD["uniform"] = {                         # sample     init        reg?
    "t-0/0/0":  "Exp-20220120-204913-OKD", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220115-093200-OKD", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220115-093239-OKD", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220120-204459-OKD", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

OKD["2018011309"] = {                      # sample     init        reg?
    "t-0/0/0":  "Exp-20220121-173212-OKD", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220115-160627-OKD", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220115-201602-OKD", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220120-204716-OKD", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

OKD["20000308"] = {                        # sample     init        reg?
    "t-0/0/0":  "Exp-20220120-204832-OKD", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20220115-232156-OKD", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20220116-091233-OKD", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20220120-205113-OKD", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

ADW["19260817"] = {                        # sample     init        reg?
    "t-0/0/0":  "Exp-20230121-114611-ADW", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20230121-145531-ADW", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20230121-160835-ADW", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20230121-134028-ADW", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

