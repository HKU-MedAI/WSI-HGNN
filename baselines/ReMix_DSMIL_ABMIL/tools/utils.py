import os
import datetime
import logging
import glob


def setup_logger(args, first_time):
    date = datetime.date.today().strftime("%Y-%m%d")
    workdir_root = 'remix_workdir/logs'
    workdir = os.path.join(f'{workdir_root}/{args.dataset}/experiments/{date}-{args.exp_name}')
    os.makedirs(workdir, exist_ok=True)
    run = len(glob.glob(os.path.join(workdir, '*.pth')))
    ckpt_pth = os.path.join(workdir, str(run + 1) + '.pth')
    if first_time:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)-4s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=f'{workdir}/run_{run}.log',
            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)-4s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(console)
    else:
        filehandler = logging.FileHandler(f'{workdir}/run_{run}.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)-4s %(message)s', '%m-%d %H:%M')
        filehandler.setFormatter(formatter)
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)
        logger.addHandler(filehandler)
        logger.setLevel(logging.DEBUG)
    return ckpt_pth
