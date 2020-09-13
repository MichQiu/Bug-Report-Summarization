import bugzilla
import pickle

def args_info_save(args):
    with open(args.args_save_path, 'wb') as f:
        pickle.dump(args, f)

class BugSource():

    def __init__(self, args, product_list):
        self.args = args
        self.url = args.url
        self.product_list = product_list
        try:
            self.finetune_ids_file = open(args.finetune_ids_file, 'r')
            self.finetune_ids = {}
            self.no_finetune_data = False
            for line in self.finetune_ids_file:
                bug_id = line[:-1]
                self.finetune_ids[bug_id] = False
            self.finetune_ids_file.close()
        except:
            self.no_finetune_data = True
        self.bzapi = bugzilla.Bugzilla(self.url)

    '''
    def source(self, bug_ids):
        """
        Obtain a dictionary that contains every product in the platform as its keys
        Within each product, there is a dictionary of bug ids and their associated bug comments
        :return {'product_A': {bug_id1:[src_text1], ...}, ...}
        """
        logger.info('Processing products...')
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(_source, self.product_list):
            product, bug_comments = d
            save_file = pjoin(self.args.save_path, product+'_bert.pt')
            logger.info('Saving to %s' % save_file)
            torch.save(bug_comments, save_file)
        pool.close()
        pool.join()
        logger.info('Processed products')

    def remove_empty_products(self):
        """Remove all product categories with 0 bug reports"""
        logger.info("Calculating number of bugs in platform products...")
        full_bugreport_len = self._calculate_bugs_len()
        logger.info("Calculation complete")
        bugreport_len = deepcopy(full_bugreport_len)
        for product in bugreport_len:
            if bugreport_len[product] == 0:
                del bugreport_len[product]
        logger.info("Removed all product categories with 0 bugs")
        return bugreport_len
    '''