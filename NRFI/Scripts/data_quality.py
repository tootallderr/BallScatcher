"""
Data quality tracking module for NRFI prediction.
Tracks and reports missing files, defaulted values, and other data quality issues.
"""
import os

class DataQualityTracker:
    """Class to track data quality metrics throughout the pipeline"""
    
    def __init__(self, threshold=0.15):
        """Initialize the tracker with a warning threshold"""
        self.threshold = threshold
        self.missing_files = set()
        self.default_counts = {}
        self.total_counts = {}
        self.groups = {}
    
    def track_missing_file(self, file_path, group=None):
        """Track a missing data file"""
        self.missing_files.add(file_path)
        
        # Also track as a default value for reporting
        file_name = os.path.basename(file_path)
        self._track_default(f"missing_file:{file_name}", 1, 1, group or "missing_files")
    
    def track_default(self, feature_name, default_count, total_count, group=None):
        """Track use of default values"""
        self._track_default(feature_name, default_count, total_count, group or "defaults")
    
    def _track_default(self, name, count, total, group):
        """Internal helper to track defaults"""
        if name not in self.default_counts:
            self.default_counts[name] = 0
            self.total_counts[name] = 0
        
        self.default_counts[name] += count
        self.total_counts[name] += total
        
        if group not in self.groups:
            self.groups[group] = set()
        self.groups[group].add(name)
    
    def get_default_rate(self, feature=None, group=None):
        """Get the default rate for a feature or group"""
        if feature:
            if feature not in self.total_counts or self.total_counts[feature] == 0:
                return 0
            return self.default_counts[feature] / self.total_counts[feature]
        
        if group:
            if group not in self.groups:
                return 0
            
            features = self.groups[group]
            defaults_sum = sum(self.default_counts.get(f, 0) for f in features)
            totals_sum = sum(self.total_counts.get(f, 0) for f in features)
            
            if totals_sum == 0:
                return 0
            return defaults_sum / totals_sum
        
        # Overall rate
        defaults_sum = sum(self.default_counts.values())
        totals_sum = sum(self.total_counts.values())
        
        if totals_sum == 0:
            return 0
        return defaults_sum / totals_sum
    
    def print_report(self):
        """Print a comprehensive data quality report"""
        print("\n=== DATA QUALITY REPORT ===")
        
        # Report missing files
        if self.missing_files:
            print(f"\nMissing Files ({len(self.missing_files)}):")
            for file_path in sorted(self.missing_files):
                print(f"  - {file_path}")
        
        # Report by group
        for group in sorted(self.groups.keys()):
            group_rate = self.get_default_rate(group=group)
            if group_rate > 0:
                print(f"\n[{group}] Default rate: {group_rate:.2%}")
                
                if group_rate > self.threshold:
                    print(f"  WARNING: Default rate exceeds threshold of {self.threshold:.2%}")
                
                # Print individual features in this group
                for feature in sorted(self.groups[group]):
                    rate = self.get_default_rate(feature=feature)
                    if rate > 0:
                        count = self.default_counts.get(feature, 0)
                        total = self.total_counts.get(feature, 0)
                        print(f"  - {feature}: {rate:.2%} ({count}/{total})")
        
        # Overall statistics
        overall_rate = self.get_default_rate()
        if overall_rate > 0:
            print(f"\nOverall Default Rate: {overall_rate:.2%}")
            
            if overall_rate > self.threshold:
                print(f"WARNING: Overall default rate exceeds threshold ({self.threshold:.2%})")
    
    def reset(self):
        """Reset all tracking data"""
        self.missing_files = set()
        self.default_counts = {}
        self.total_counts = {}
        self.groups = {}
    
    def get_missing_files_count(self):
        """Get count of missing files"""
        return len(self.missing_files)

# Create a global instance for convenience
quality_tracker = DataQualityTracker()

# Convenience functions
def track_missing_file(file_path, group=None):
    quality_tracker.track_missing_file(file_path, group)

def track_default(feature_name, default_count, total_count, group=None):
    quality_tracker.track_default(feature_name, default_count, total_count, group)

def print_quality_report():
    quality_tracker.print_report()

def reset_quality_tracker():
    quality_tracker.reset()
