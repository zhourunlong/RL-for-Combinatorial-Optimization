SP = {}
OKD = {}
ADW = {}

PROBLEMS = {"SP": SP, "OKD": OKD, "ADW": ADW}

ADW["19260817-uniform"] = {                # sample     init        reg?
    "t-0/0/0":  "Exp-20230121-114611-ADW", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20230121-145531-ADW", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20230121-160835-ADW", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20230121-134028-ADW", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

ADW["19260817-special"] = {                # sample     init        reg?
    "t-0/0/0":  "Exp-20230126-012751-ADW", # pi^t-pi^0	pi^0        0
    "t-0/0/r":  None,                      # pi^t-pi^0	pi^0	    reg
    "t/0/0":    "Exp-20230125-222152-ADW", # pi^t	    pi^0        0
    "t/0/r":    None,                      # pi^t	    pi^0        reg
    "0/0/0":    "Exp-20230126-040913-ADW", # pi^0	    pi^0        0
    "0/0/r":    None,                      # pi^0	    pi^0        reg
    "t/0-t/0":  "Exp-20230125-235911-ADW", # pi^t	    pi^0-pi^t   0
    "t/0-t/r":  None,                      # pi^t	    pi^0-pi^t   reg
}

