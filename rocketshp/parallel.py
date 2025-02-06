from multiprocessing import Pool
from itertools import count, chain
from loguru import logger
import math
from tqdm import tqdm

def create_batches(dataset, batch_size):
    """Create batches from a list of IDs."""
    dataset_size = len(dataset)
    for i in range(0, dataset_size, batch_size):
        yield [dataset[i] for i in range(i, min(i + batch_size, dataset_size))]

def process_batch_wrapper(args):
    """Wrapper to handle job numbering and batch processing."""
    job_num, batch, process_fn, report_every = args
    
    if report_every:
        logger.info(f"Job {job_num}: Processing batch of size {len(batch)}")
    
    try:
        result = process_fn(batch)
        
        if report_every and job_num % report_every == 0:
            logger.info(f"Job {job_num}: Completed successfully")
            
        return result
    
    except Exception as e:
        logger.info(f"Job {job_num}: Failed with error: {str(e)}")
        return []

def parallel_pool(dataset, process_fn, n_jobs=4, batch_size=32, report_every=None):
    """
    Process an h5 file in parallel using a process pool.
    
    Args:
        dataset: List of IDs
        process_fn: Function that takes a batch of IDs and returns a list of results
        n_jobs: Number of parallel jobs to run
        batch_size: Size of batches to process
        report_every: How often to report progress (None for no reporting)
    
    Returns:
        List of results from all batches, flattened
    """
    total_batches = math.ceil(len(dataset) / batch_size)
    batches = create_batches(dataset, batch_size)
    
    # Create tuples of (job_num, batch, process_fn, report_every)
    job_args = ((i, batch, process_fn, report_every) 
                for i, batch in zip(count(), batches))
    
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_batch_wrapper, job_args),
            total=total_batches,
            desc="Processing batches"
        ))
    
    # Flatten results
    return list(chain.from_iterable(results))