"""
DefaultTracker: Monitors and reports usage of default values in data pipelines
"""
import os

class DefaultTracker:
    def __init__(self, threshold=0.15):
        """
        Initialize the default tracker
        
        Parameters:
        -----------
        threshold : float
            Warning threshold for defaulted values (as proportion)
        """
        self.default_counts = {}  # {feature_name: count}
        self.total_counts = {}    # {feature_name: total}
        self.threshold = threshold
        self.feature_groups = {}  # {group_name: [feature_names]}
        self.missing_files = set()  # Track missing files
        
    def reset(self):
        """Reset all tracking data"""
        self.default_counts = {}
        self.total_counts = {}
        self.feature_groups = {}
        self.missing_files = set()
        
    def track_default(self, feature_name, default_count, total_count, group=None):
        """
        Track default value usage
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature being tracked
        default_count : int
            Number of values defaulted
        total_count : int
            Total number of values
        group : str, optional
            Group name to organize features (e.g., 'pitcher_stats')
        """
        if feature_name not in self.default_counts:
            self.default_counts[feature_name] = 0
            self.total_counts[feature_name] = 0
            
        self.default_counts[feature_name] += default_count
        self.total_counts[feature_name] += total_count
        
        # Track feature in group
        if group:
            if group not in self.feature_groups:
                self.feature_groups[group] = set()
            self.feature_groups[group].add(feature_name)
    
    def track_missing_file(self, file_path, group=None):
        """
        Track a completely missing file
        
        Parameters:
        -----------
        file_path : str
            Path to the missing file
        group : str, optional
            Group name to organize features
        """
        file_name = os.path.basename(file_path)
        feature_name = f"missing_file:{file_name}"
        self.default_counts[feature_name] = 1
        self.total_counts[feature_name] = 1
        self.missing_files.add(file_path)
        
        # Track feature in group
        if group:
            if group not in self.feature_groups:
                self.feature_groups[group] = set()
            self.feature_groups[group].add(feature_name)
    
    def get_default_rate(self, feature_name):
        """Get the default rate for a specific feature"""
        if feature_name not in self.total_counts or self.total_counts[feature_name] == 0:
            return 0
        return self.default_counts[feature_name] / self.total_counts[feature_name]
    
    def get_group_default_rate(self, group):
        """Get the overall default rate for a group"""
        if group not in self.feature_groups:
            return 0
            
        total_defaults = 0
        total_counts = 0
        
        for feature in self.feature_groups[group]:
            total_defaults += self.default_counts.get(feature, 0)
            total_counts += self.total_counts.get(feature, 0)
            
        if total_counts == 0:
            return 0
        return total_defaults / total_counts
    
    def get_overall_default_rate(self):
        """Get the overall default rate across all features"""
        total_defaults = sum(self.default_counts.values())
        total_counts = sum(self.total_counts.values())
        
        if total_counts == 0:
            return 0
        return total_defaults / total_counts
    
    def print_report(self):
        """Print a summary report of default usage"""
        print("\n=== DEFAULT VALUE USAGE REPORT ===")
        
        # Report missing files
        if self.missing_files:
            print(f"\nMissing Files ({len(self.missing_files)}):")
            for file_path in sorted(self.missing_files):
                print(f"  - {file_path}")
        
        # Report by group
        for group in sorted(self.feature_groups.keys()):
            group_defaults = sum(self.default_counts.get(feature, 0) 
                               for feature in self.feature_groups[group])
            group_total = sum(self.total_counts.get(feature, 0) 
                             for feature in self.feature_groups[group])
            
            if group_total > 0:
                group_rate = group_defaults / group_total
                print(f"\n[{group}] Default rate: {group_rate:.2%}")
                
                if group_rate > self.threshold:
                    print(f"  WARNING: Default rate exceeds threshold of {self.threshold:.2%}")
                
                # Show individual features
                for feature in sorted(self.feature_groups[group]):
                    default_rate = self.get_default_rate(feature)
                    if default_rate > 0:
                        defaults = self.default_counts.get(feature, 0)
                        total = self.total_counts.get(feature, 0)
                        print(f"  - {feature}: {default_rate:.2%} ({defaults}/{total})")
        
        # Overall statistics
        total_defaults = sum(self.default_counts.values())
        total_values = sum(self.total_counts.values())
        
        if total_values > 0:
            overall_rate = total_defaults / total_values
            print(f"\nOverall Default Rate: {overall_rate:.2%}")
            
            if overall_rate > self.threshold:
                print(f"WARNING: Overall default rate exceeds threshold ({self.threshold:.2%})")

    def summarize_data_quality(self):
        """Generate a summary of data quality"""
        summary = []
        summary.append("=== DATA QUALITY SUMMARY ===")
        
        # Missing files summary
        missing_files_count = len(self.missing_files)
        if missing_files_count > 0:
            summary.append(f"\nMissing Files ({missing_files_count}):")
            for file in sorted(list(self.missing_files)):
                summary.append(f"  - {file}")
        
        # Groups with excessive defaults
        excessive_groups = []
        for group in sorted(self.feature_groups.keys()):
            rate = self.get_group_default_rate(group)
            if rate > self.threshold:
                excessive_groups.append((group, rate))
        
        if excessive_groups:
            summary.append("\nGroups with High Default Rates:")
            for group, rate in sorted(excessive_groups, key=lambda x: x[1], reverse=True):
                summary.append(f"  - {group}: {rate:.2%}")
        
        # Overall stats
        overall_rate = self.get_overall_default_rate()
        summary.append(f"\nOverall Default Rate: {overall_rate:.2%}")
        if overall_rate > self.threshold:
            summary.append(f"WARNING: Overall default rate exceeds threshold ({self.threshold:.2%})")
            
        # Add recommendations if quality is poor
        if overall_rate > self.threshold or missing_files_count > 0:
            summary.append("\nRecommendations:")
            if missing_files_count > 0:
                summary.append("  - Locate and provide missing data files")
            if excessive_groups:
                for group, _ in excessive_groups:
                    summary.append(f"  - Improve data quality for {group}")
        
        return "\n".join(summary)

# Create a global instance
default_tracker = DefaultTracker(threshold=0.15)
