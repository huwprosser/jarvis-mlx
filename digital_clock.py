#!/usr/bin/env python3
"""
Digital Clock - Displays current time in multiple time zones
A simple yet elegant multi-timezone clock application
"""

from datetime import datetime
import pytz
from typing import List, Dict
import os
import sys

class DigitalClock:
    """A digital clock that displays time in multiple time zones"""
    
    def __init__(self, timezones: List[str] = None):
        """
        Initialize the digital clock with specified timezones
        
        Args:
            timezones: List of timezone strings (e.g., ['US/Eastern', 'Europe/London'])
                      If None, uses default major timezones
        """
        if timezones is None:
            self.timezones = [
                'US/Eastern',
                'US/Central',
                'US/Mountain',
                'US/Pacific',
                'Europe/London',
                'Europe/Paris',
                'Asia/Tokyo',
                'Australia/Sydney'
            ]
        else:
            self.timezones = timezones
        
        # Validate timezones
        self._validate_timezones()
    
    def _validate_timezones(self):
        """Validate that all specified timezones are valid"""
        valid_zones = pytz.all_timezones
        for tz in self.timezones:
            if tz not in valid_zones:
                raise ValueError(f"Invalid timezone: {tz}")
    
    def get_time_in_timezone(self, timezone_str: str) -> datetime:
        """
        Get current time in a specific timezone
        
        Args:
            timezone_str: Timezone string (e.g., 'US/Eastern')
            
        Returns:
            datetime object with timezone information
        """
        tz = pytz.timezone(timezone_str)
        return datetime.now(tz)
    
    def get_all_times(self) -> Dict[str, str]:
        """
        Get current time in all configured timezones
        
        Returns:
            Dictionary with timezone names as keys and formatted time strings as values
        """
        times = {}
        for tz_str in self.timezones:
            dt = self.get_time_in_timezone(tz_str)
            times[tz_str] = dt.strftime('%H:%M:%S %Z')
        return times
    
    def display_clock(self, clear_screen: bool = True):
        """
        Display the digital clock in the terminal
        
        Args:
            clear_screen: Whether to clear the screen before displaying
        """
        if clear_screen and sys.platform != 'win32':
            os.system('clear')
        elif clear_screen and sys.platform == 'win32':
            os.system('cls')
        
        print("\n" + "="*60)
        print(" " * 15 + "⏰ DIGITAL CLOCK ⏰")
        print("="*60 + "\n")
        
        times = self.get_all_times()
        
        for timezone, time_str in times.items():
            # Format timezone name nicely
            display_name = timezone.replace('_', ' ').ljust(20)
            print(f"  {display_name} │ {time_str}")
        
        print("\n" + "="*60 + "\n")
    
    def display_specific_timezone(self, timezone_str: str):
        """
        Display time for a specific timezone
        
        Args:
            timezone_str: Timezone string (e.g., 'US/Eastern')
        """
        dt = self.get_time_in_timezone(timezone_str)
        display_name = timezone_str.replace('_', ' ')
        time_str = dt.strftime('%H:%M:%S %Z')
        print(f"{display_name}: {time_str}")
    
    def get_24hr_format(self) -> Dict[str, str]:
        """Get times in 24-hour format with detailed info"""
        times = {}
        for tz_str in self.timezones:
            dt = self.get_time_in_timezone(tz_str)
            times[tz_str] = {
                'time': dt.strftime('%H:%M:%S'),
                'date': dt.strftime('%Y-%m-%d'),
                'full': dt.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')
            }
        return times
    
    def get_12hr_format(self) -> Dict[str, str]:
        """Get times in 12-hour format with AM/PM"""
        times = {}
        for tz_str in self.timezones:
            dt = self.get_time_in_timezone(tz_str)
            times[tz_str] = dt.strftime('%I:%M:%S %p %Z')
        return times
    
    @staticmethod
    def get_available_timezones() -> List[str]:
        """Get list of all available timezones"""
        return pytz.all_timezones
    
    @staticmethod
    def list_timezones_by_region(region: str) -> List[str]:
        """
        List all timezones in a specific region
        
        Args:
            region: Region name (e.g., 'US', 'Europe', 'Asia')
            
        Returns:
            List of timezones matching the region
        """
        return [tz for tz in pytz.all_timezones if region in tz]


def main():
    """Main entry point for the digital clock"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Digital Clock - Multi-timezone time display')
    parser.add_argument('--timezones', nargs='+', help='Specific timezones to display')
    parser.add_argument('--timezone', help='Display time in a specific timezone')
    parser.add_argument('--list-regions', help='List timezones in a region')
    parser.add_argument('--format', choices=['24hr', '12hr'], default='24hr', help='Time format')
    parser.add_argument('--continuous', action='store_true', help='Update clock continuously')
    
    args = parser.parse_args()
    
    # Handle listing timezones by region
    if args.list_regions:
        timezones = DigitalClock.list_timezones_by_region(args.list_regions)
        print(f"\nTimezones in '{args.list_regions}':")
        for tz in timezones:
            print(f"  - {tz}")
        print()
        return
    
    # Create clock instance
    if args.timezones:
        clock = DigitalClock(args.timezones)
    else:
        clock = DigitalClock()
    
    # Display specific timezone
    if args.timezone:
        clock.display_specific_timezone(args.timezone)
        return
    
    # Display clock
    if args.continuous:
        try:
            import time
            while True:
                clock.display_clock(clear_screen=True)
                if args.format == '12hr':
                    times = clock.get_12hr_format()
                else:
                    times = clock.get_24hr_format()
                    for tz, time_info in times.items():
                        print(f"  {tz}: {time_info['full']}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nClock stopped.")
    else:
        if args.format == '12hr':
            times = clock.get_12hr_format()
            print("\n" + "="*60)
            print(" " * 15 + "⏰ DIGITAL CLOCK ⏰")
            print("="*60 + "\n")
            for tz, time_str in times.items():
                display_name = tz.replace('_', ' ').ljust(20)
                print(f"  {display_name} │ {time_str}")
        else:
            times = clock.get_24hr_format()
            print("\n" + "="*60)
            print(" " * 15 + "⏰ DIGITAL CLOCK ⏰")
            print("="*60 + "\n")
            for tz, time_info in times.items():
                display_name = tz.replace('_', ' ').ljust(20)
                print(f"  {display_name} │ {time_info['full']}")
        print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
